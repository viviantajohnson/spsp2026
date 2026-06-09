# SEMANTIC FRAME ANALYSIS USING SENTENCE EMBEDDINGS

# PURPOSE: Assigns theoretically defined frames to social media posts
# using semantic similarity. It embeds both the frame definitions and the
# texts into the same semantic vector space.


# FRAME ASSIGNMENT LOGIC:

#1. Convert each frame definition into an embedding.
#2. Convert each tweet/post into an embedding.
#3. Calculate cosine similarity between each post and each frame.
#4. Assign the most similar frame(s).
#5. Save both categorical frame labels and continuous similarity scores.

# This approach captures semantic meaning rather than exact word matching.
#  e.g., "this is dangerous" and "this poses a serious risk" may be
# classified similarly even if they share few keywords.

###############################################################################

# PACKAGE SETUP

# Install required packages
# Note: installation only needs to be performed once per machine.

install.packages(c("dplyr", "stringr", "lubridate", "readr"))
install.packages(c("reticulate", "data.table"))
install.packages(c("lme4", "lmerTest"))

# Load required packages

library(dplyr)
library(stringr)
library(lubridate)
library(reticulate)
library(data.table)
library(lme4)
library(lmerTest)
library(emmeans)

options(scipen = 999) # Turn off scientific notation

# Uplolad dataset and name it 'df'

df <- as.data.frame(shortdf)

###############################################################################

# STEP 1: PREPARE TEXT DATA

df <- df %>%
  mutate(
    embed_text = cleanedtext, # embed_text = text that will be embedded by
                              # sentence-transformer model.
    month = floor_date(extractedts, "month") # month = month-level time
                                              # variable
  ) %>%
  filter(!is.na(embed_text), # filter = removes missing texts 
         nchar(str_trim(embed_text)) >= 3) # and short texts (< 3 char.)
            

###############################################################################

# STEP 2: DEFINE THEORETICAL FRAME CODEBOOK

frames <- tibble::tribble(
  ~frame, ~description,
  "moral_judgment",      "Language that frames events as morally wrong or
                          unjust, assigns blame or responsibility, emphasizes
                          violations of norms or law, and calls for
                          accountability, punishment, or condemnation",
  "threat_appraisal",    "Language that emphasizes danger, escalation, or
                          catastrophic risk, highlighting threats to safety,
                          stability, or survival and projecting potential
                          future harm.",
  "identity_signaling",  "Language that signals group alignment or coalition
                          membership, emphasizes shared values or collective
                          identity, and distinguishes between ingroups and
                          outgroups.",
  "care_harm_concerns",  "Language that emphasizes suffering, humanitarian
                          concern, empathy, or the need to protect and assist
                          vulnerable individuals or communities."
)

###############################################################################

# STEP 3: LOAD PYTHON SENTENCE-TRANSFORMER MODEL

py_install(packages = c("sentence-transformers",
                        "torch",
                        "numpy"),pip = TRUE)

st <- import("sentence_transformers")
np <- import("numpy")

model <- st$SentenceTransformer("all-MiniLM-L6-v2") # widely used sentence
                                                    # embedding model from
                                                    # Hugging Face

# Output: Each text becomes a 384-dimensional vector representation.
# Similar meanings should occupy nearby locations in this vector space.

###############################################################################

# STEP 4: EMBED FRAME DEFINITIONS
# This is the semantic dictionary; one embedding created per frame.

# Result:
# frame_emb = matrix with dimensions:
# number_of_frames x 384

# Example:
# moral_judgment -> vector
# threat_appraisal -> vector
# identity_signaling -> vector
# care_harm_concerns -> vector

# These vectors serve as semantic reference points against which all posts are
# compared.

frame_emb <- model$encode(
  frames$description,
  batch_size = as.integer(32),
  show_progress_bar = FALSE,
  convert_to_numpy = TRUE,
  normalize_embeddings = TRUE
) |> py_to_r()  # K x 384

dim(frame_emb)

###############################################################################

# STEP 5: FRAME ASSIGNMENT

# INPUT:
# emb_block = embeddings for a subset of posts
# frame_emb = embeddings for frame definitions
# frame_labels = frame names

# OUTPUT:
# frame1 = most similar frame
# frame2 = second most similar frame
# frame1_sim = similarity score for frame1
# frame2_sim = similarity score for frame2
# sim_df = similarity scores for ALL frames

# NOTE: Because embeddings are normalized, matrix multiplication is
# equivalent to cosine similarity.

# Interpretation:
# Higher values -> stronger semantic alignment
# Lower values -> weaker semantic alignment

# Similarity scores are continuous measures and often provide more info
# than categorical frame assignments.
assign_frames_top2 <- function(emb_block, frame_emb,
                               frame_labels, min_sim = 0.20) {
  
# Compute cosine similarity between every post and every frame.
# Output dimensions: rows = posts; columns = frames
  
  sims <- emb_block %*% t(frame_emb)
  
# Save all frame similarities.
# Example:
# sim_moral_judgment
# sim_threat_appraisal
# sim_identity_signaling
# sim_care_harm_concerns

  sim_df <- as.data.frame(sims)
  colnames(sim_df) <- paste0("sim_", frame_labels)
  
# Identify highest-scoring frame

  k1 <- max.col(sims, ties.method = "first")

# Extract similarity score associated with highest-scoring frame.
  
  s1 <- sims[cbind(seq_len(nrow(sims)), k1)]

# Remove top frame and identify second-highest frame.
  
  sims2 <- sims
  sims2[cbind(seq_len(nrow(sims2)), k1)] <- -Inf
  
  k2 <- max.col(sims2, ties.method = "first")
  s2 <- sims2[cbind(seq_len(nrow(sims2)), k2)]

  f1 <- frame_labels[k1]
  f2 <- frame_labels[k2]

# Apply minimum similarity threshold. Posts below threshold are
# considered insufficiently similar to any frame. Threshold should ideally be validated through human coding or sensitivity
# analyses. 
  
f1[s1 < min_sim] <- "unclassified"
f2[s1 < min_sim] <- "unclassified"
  
  list(
    frame1 = f1,
    frame1_sim = s1,
    frame2 = f2,
    frame2_sim = s2,
    sim_df = sim_df
  )
}

# Step 6: Run chunked embedding & save ALL frame similarity scores

# Create columns for top frames

df$frame1 <- NA_character_
df$frame1_sim <- NA_real_
df$frame2 <- NA_character_
df$frame2_sim <- NA_real_

# Create columns for every frame similarity score

for(f in frames$frame){
  df[[paste0("sim_", f)]] <- NA_real_
}

frame_labels <- frames$frame

block_size <- 50000
starts <- seq(1, nrow(df), by = block_size)

for (s in starts) {
  
  e <- min(s + block_size - 1, nrow(df))
  
  cat("Framing rows", s, "to", e, "\n")
  
  emb_block <- model$encode(
    df$embed_text[s:e],
    batch_size = as.integer(128),
    show_progress_bar = FALSE,
    convert_to_numpy = TRUE,
    normalize_embeddings = TRUE
  ) |> py_to_r()
  
  out <- assign_frames_top2(
    emb_block,
    frame_emb,
    frame_labels,
    min_sim = 0.20
  )
  
# Save top frame assignments
  
  df$frame1[s:e] <- out$frame1
  df$frame1_sim[s:e] <- out$frame1_sim
  
  df$frame2[s:e] <- out$frame2
  df$frame2_sim[s:e] <- out$frame2_sim
  
# Save ALL similarity scores
  
  sim_cols <- colnames(out$sim_df)
  
  for(col in sim_cols){
    df[s:e, col] <- out$sim_df[[col]]
  }
  
}

# View df

head(df)

################################################################################

# INTERPRETING OUTPUT VARIABLES

# frame1: Highest-scoring frame.
# frame2: Second-highest-scoring frame.
# frame1_sim: Similarity score for frame1.
# frame2_sim: Similarity score for frame2.
# sim_moral_judgment: Continuous similarity to moral judgment frame.
# sim_threat_appraisal: Continuous similarity to threat frame.
# sim_identity_signaling: Continuous similarity to identity frame.
# sim_care_harm_concerns: Continuous similarity to care/harm frame.

# Note: For inferential analyses, use continuous similarity scores
# instead of categorical frame labels. Continuous scores preserve more
# info about the strength of framing.

