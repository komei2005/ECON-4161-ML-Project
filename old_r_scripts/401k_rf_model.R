# 401(k) Treatment Effect -> Random Forest Model
# Komei, Kai, Sheriloye, Toye
#libraries to be imported
library(tidyverse)
library(caret)
library(ranger) # a random forest library, so the technique is still random forest

#for reproducibility
set.seed(42)

#this is the cleaned dataset we now have from our data cleaning process
data <- read.csv("401k_clean.csv")


# Run in console only - do not add to script
# str(data)
# summary(data)
# head(data)

# Predictor set for the wealth model.
# Excluded for leakage: a401, tfa, net_tfa, tfa_he, ira, hval, hmort, hequity, nifa, net_nifa, net_n401
#   (these columns mechanically contain 401(k) assets and would leak the treatment into the outcome)
# Excluded as redundant for Random Forest: age_sq, inc_k, inc_k_sq, educ_sq, fsize_sq,
#   educ_group, age_group, inc_group (trees handle nonlinearities natively; the squared/group
#   columns are reserved for the OLS/2SLS replication and heterogeneity analysis)
predictors <- c("age", "inc", "fsize", "educ", "db", "marr", "male",
                "twoearn", "pira", "hown")

# tw = total wealth (outcome); p401 = 401(k) participation indicator (treatment)
outcome <- "tw"
treatment <- "p401"

# Build a clean modeling dataframe with outcome, treatment, and predictors only.
# All downstream steps (train/test split, RF training, counterfactual predictions)
# operate on model_data rather than the full data object.
model_data <- data[, c(outcome, treatment, predictors)]

# Stratified 70/30 train/test split on the treatment variable (p401).
# Stratification preserves the ~26%/74% participation mix in both sets, which
# stabilizes the treatment-effect estimate by avoiding an unlucky split that
# leaves too few treated observations in the test set.
# The test set is held out: not used for hyperparameter tuning, only for final
# predictive evaluation and treatment-effect estimation.
train_index <- createDataPartition(model_data[[treatment]], p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]


# Actual model
# Treatment included as a feature so the S-learner can later predict
# counterfactual outcomes by flipping p401.
features <- c(treatment, predictors)

# 10-fold cross-validation on the training set. More folds than the previous
# 5-fold pass to get more stable hyperparameter selection. The test set
# remains untouched throughout tuning.
cv_control <- trainControl(method = "cv", number = 10)

# Expanded hyperparameter grid covering three Random Forest knobs:
# - mtry: number of features randomly sampled at each split (most important).
# - splitrule: "variance" is standard RF; "extratrees" adds extra split
#   randomization that often improves out-of-sample performance.
# - min.node.size: minimum size of terminal nodes (smaller = deeper trees =
#   more flexibility, with a mild overfitting risk that CV controls for).
tune_grid <- expand.grid(
  mtry = c(2, 3, 4, 5, 6, 8, 10),
  splitrule = c("variance", "extratrees"),
  min.node.size = c(5, 10, 20)
)

# Train the Random Forest with ranger (a fast RF implementation) via caret.
# ntree fixed at 500 (no overfitting risk from more trees, and 500 is plenty
# for a dataset of this size). Tuning is over the three hyperparameters above.
rf_tuned <- train(
  x = train_data[, features],
  y = train_data[[outcome]],
  method = "ranger",
  trControl = cv_control,
  tuneGrid = tune_grid,
  num.trees = 500
)

# Predict on the held-out test set using the best hyperparameter combination
# selected by cross-validation.
test_predictions <- predict(rf_tuned, newdata = test_data[, features])

# Test-set RMSE and R-squared.
# Reported as the honest measure of out-of-sample predictive performance.
test_rmse <- sqrt(mean((test_data[[outcome]] - test_predictions)^2))
test_r2 <- 1 - sum((test_data[[outcome]] - test_predictions)^2) /
  sum((test_data[[outcome]] - mean(test_data[[outcome]]))^2)

# Run these in the terminal to check
# print(rf_tuned)
# test_rmse
# test_r2

# S-learner counterfactual prediction.
# For each test observation, predict wealth twice using the trained Random Forest:
# once with p401 forced to 1 (treated counterfactual) and once with p401 forced
# to 0 (control counterfactual). The per-row difference is the individual
# treatment effect; averaging across the test set gives the ATE.
test_treated <- test_data
test_treated[[treatment]] <- 1

test_control <- test_data
test_control[[treatment]] <- 0

# Predict total wealth under each counterfactual scenario.
# rf_tuned is the cross-validated Random Forest trained earlier on the training set.
pred_treated <- predict(rf_tuned, newdata = test_treated[, features])
pred_control <- predict(rf_tuned, newdata = test_control[, features])

# Individual treatment effects: one estimate per test observation.
# Saved for later use in the heterogeneity analysis (subgroup means by income,
# education, etc.).
ite <- pred_treated - pred_control

# Average treatment effect: the headline causal estimate from the ML model.
# Reported in dollars; directly comparable to the OLS and 2SLS coefficients
# from the Chernozhukov & Hansen (2004) Panel A replication.
ate <- mean(ite)

# Subgroup-level out-of-sample RMSE for diagnostic purposes.
# pred_treated and pred_control are counterfactual predictions, so a global
# comparison against test_data$tw mixes factual and counterfactual cases.
# The meaningful diagnostic is: how well does the model predict for actually
# treated households (using pred_treated) and actually control households
# (using pred_control), separately. Comparable subgroup RMSEs indicate the
# model performs evenly across treatment status; a large gap would flag
# systematic bias toward one group.
treated_idx <- test_data$p401 == 1
control_idx <- test_data$p401 == 0

rmse_treated_subgroup <- sqrt(mean(
  (test_data$tw[treated_idx] - pred_treated[treated_idx])^2
))

rmse_control_subgroup <- sqrt(mean(
  (test_data$tw[control_idx] - pred_control[control_idx])^2
))

# Sample sizes for each subgroup, useful for interpreting the RMSE difference
# (treated subgroup is roughly 26% of the test set, control roughly 74%).
n_treated <- sum(treated_idx)
n_control <- sum(control_idx)

# Run these in the console to check
# rmse_treated_subgroup
# rmse_control_subgroup
# n_treated
# n_control

# Save artifacts for handoff to the analysis team (Renkai and Komei).
# Three files: the trained model, the test set with individual treatment effects
# attached, and a summary of headline numbers. With these, the analysis team
# can run heterogeneity analysis and the OLS/2SLS comparison without re-running
# the full pipeline.

# 1. Trained Random Forest model object.
# Load on the analysis side with: rf_tuned <- readRDS("rf_model.rds")
saveRDS(rf_tuned, "rf_model.rds")

# 2. Test set joined with per-observation S-learner outputs.
# Each row is one test observation. Includes the original outcome (tw), the
# treatment indicator (p401), all predictors, and three new columns:
#   - pred_treated: model prediction with p401 = 1
#   - pred_control: model prediction with p401 = 0
#   - ite:          individual treatment effect (pred_treated - pred_control)
#   - row_id:       original row index in the cleaned dataset (401k_clean.csv),
#                   so the analysis team can join with auxiliary columns
#                   (educ_group, age_group, inc_group) for heterogeneity slicing.
test_results <- test_data
test_results$pred_treated <- pred_treated
test_results$pred_control <- pred_control
test_results$ite <- ite
test_results$row_id <- as.integer(rownames(test_data))
write.csv(test_results, "test_results.csv", row.names = FALSE)

# 3. Headline model summary as a small two-column CSV.
model_summary <- data.frame(
  metric = c("Test RMSE", "Test R-squared", "Average Treatment Effect"),
  value  = c(test_rmse, test_r2, ate)
)
write.csv(model_summary, "model_summary.csv", row.names = FALSE)

