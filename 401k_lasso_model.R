# 401(k) Treatment Effect -> LASSO Model
# Komei, Kai, Sheriloye, Toye
# libraries to be imported
library(tidyverse)
library(caret)
library(glmnet) # penalized linear models including LASSO

# for reproducibility
set.seed(42)

# this is the cleaned dataset we now have from our data cleaning process
data <- read.csv("401k_clean.csv")

# Predictor set for the wealth model.
# Excluded: a401, tfa, net_tfa, tfa_he, ira, hval, hmort, hequity, nifa, net_nifa, net_n401
predictors <- c("age", "inc", "fsize", "educ", "db", "marr", "male",
                "twoearn", "pira", "hown")

outcome <- "tw"
treatment <- "p401"

# every possible pairwise interaction among treatment and predictors
add_engineered_terms <- function(df, base_vars) {
  interaction_pairs <- combn(base_vars, 2, simplify = FALSE)
  for (pair in interaction_pairs) {
    interaction_name <- paste0(pair[1], "_x_", pair[2])
    df[[interaction_name]] <- df[[pair[1]]] * df[[pair[2]]]
  }
  df
}

# build modeling dataframe and add all pairwise interactions.
base_vars <- c(treatment, predictors)
model_data <- add_engineered_terms(data[, c(outcome, base_vars)], base_vars)

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
# Include treatment, base controls, and all pairwise interactions.
interaction_features <- setdiff(names(model_data), c(outcome, base_vars))
features <- c(base_vars, interaction_features)

# 10-fold cross-validation on the training set.
cv_control <- trainControl(method = "cv", number = 10)

# LASSO tuning
# lambda controls shrinkage strength
lasso_grid <- expand.grid(
  alpha = 1,
  lambda = 10^seq(-3, 1.5, length.out = 120)
)

# train LASSO
lasso_tuned <- train(
  x = train_data[, features],
  y = train_data[[outcome]],
  method = "glmnet",
  trControl = cv_control,
  tuneGrid = lasso_grid,
  preProcess = c("center", "scale")
)

# extract coefficients at the CV-selected lambda
coef_matrix <- as.matrix(coef(
  lasso_tuned$finalModel,
  s = lasso_tuned$bestTune$lambda
))

coef_table <- data.frame(
  term = rownames(coef_matrix),
  coefficient = as.numeric(coef_matrix[, 1]),
  is_interaction = grepl("_x_", rownames(coef_matrix)),
  stringsAsFactors = FALSE
)

interaction_coef_table <- coef_table[coef_table$is_interaction, ]

# predict on test set using the best lambda selected by CV.
test_predictions <- predict(lasso_tuned, newdata = test_data[, features])

# test-set RMSE and R-squared.
test_rmse <- sqrt(mean((test_data[[outcome]] - test_predictions)^2))
test_r2 <- 1 - sum((test_data[[outcome]] - test_predictions)^2) /
  sum((test_data[[outcome]] - mean(test_data[[outcome]]))^2)


# S-learner counterfactual prediction.
# For each test observation, predict wealth twice using the trained LASSO:
# once with p401 forced to 1 (treated counterfactual) and once with p401 forced
# to 0 (control counterfactual). The per-row difference is the individual
# treatment effect; averaging across the test set gives the ATE.
test_treated <- test_data
test_treated[[treatment]] <- 1
test_treated <- add_engineered_terms(test_treated, base_vars)

test_control <- test_data
test_control[[treatment]] <- 0
test_control <- add_engineered_terms(test_control, base_vars)

# Predict total wealth under each counterfactual scenario.
# lasso_tuned is the cross-validated LASSO trained earlier on the training set.
pred_treated <- predict(lasso_tuned, newdata = test_treated[, features])
pred_control <- predict(lasso_tuned, newdata = test_control[, features])

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
# (using pred_control), separately.
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

# Save artifacts for handoff to the analysis team.
# Three files: the trained model, the test set with individual treatment effects
# attached, and a summary of headline numbers.

# 1. Trained LASSO model object.
# Load on the analysis side with: lasso_tuned <- readRDS("lasso_model.rds")
saveRDS(lasso_tuned, "lasso_model.rds")

# 2. Test set joined with per-observation S-learner outputs.
test_results <- test_data
test_results$pred_treated <- pred_treated
test_results$pred_control <- pred_control
test_results$ite <- ite
test_results$row_id <- as.integer(rownames(test_data))
write.csv(test_results, "lasso_test_results.csv", row.names = FALSE)

# 3. Headline model summary as a small two-column CSV.
model_summary <- data.frame(
  metric = c("Test RMSE", "Test R-squared", "Average Treatment Effect"),
  value  = c(test_rmse, test_r2, ate)
)
write.csv(model_summary, "lasso_model_summary.csv", row.names = FALSE)

# 4. Full coefficient table at selected lambda.
write.csv(coef_table, "lasso_coefficients.csv", row.names = FALSE)

# 5. Interaction-only coefficient table.
write.csv(interaction_coef_table, "lasso_interaction_coefficients.csv", row.names = FALSE)
