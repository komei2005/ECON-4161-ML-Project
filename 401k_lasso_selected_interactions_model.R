# 401(k) Treatment Effect -> LASSO Model (Selected Interactions)
# Komei, Kai, Sheriloye, Toye
# libraries to be imported
library(tidyverse)
library(caret)
library(glmnet) # penalized linear models including LASSO

# for reproducibility
set.seed(42)

# this is the cleaned dataset we now have from our data cleaning process
data <- read.csv("401k_clean.csv")


# Run in console only - do not add to script
# str(data)
# summary(data)
# head(data)

# Predictor set for the wealth model.
# Excluded for leakage: a401, tfa, net_tfa, tfa_he, ira, hval, hmort, hequity, nifa, net_nifa, net_n401
#   (these columns mechanically contain 401(k) assets and would leak the treatment into the outcome)
predictors <- c("age", "inc", "fsize", "educ", "db", "marr", "male",
                "twoearn", "pira", "hown")

# tw = total wealth (outcome); p401 = 401(k) participation indicator (treatment)
outcome <- "tw"
treatment <- "p401"

# Engineer only the requested interaction terms:
# - age x income
# - age x home ownership
# - age x IRA participation
# - income x IRA participation
# - income x home ownership
# - income x marriage
# - income x two earners
add_selected_interactions <- function(df) {
  df$age_x_inc <- df$age * df$inc
  df$age_x_hown <- df$age * df$hown
  df$age_x_pira <- df$age * df$pira
  df$inc_x_pira <- df$inc * df$pira
  df$inc_x_hown <- df$inc * df$hown
  df$inc_x_marr <- df$inc * df$marr
  df$inc_x_twoearn <- df$inc * df$twoearn
  df
}

# Build a clean modeling dataframe and add selected interactions.
model_data <- add_selected_interactions(data[, c(outcome, treatment, predictors)])

# Stratified 70/30 train/test split on the treatment variable (p401).
# Stratification preserves the participation mix in both sets.
train_index <- createDataPartition(model_data[[treatment]], p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]


# Actual model
interaction_features <- c(
  "age_x_inc", "age_x_hown", "age_x_pira",
  "inc_x_pira", "inc_x_hown", "inc_x_marr", "inc_x_twoearn"
)
features <- c(treatment, predictors, interaction_features)

# 10-fold cross-validation on the training set.
cv_control <- trainControl(method = "cv", number = 10)

# LASSO tuning grid:
# - alpha = 1 enforces pure LASSO (L1 penalty).
# - lambda controls shrinkage strength (larger = more regularization/sparsity).
lasso_grid <- expand.grid(
  alpha = 1,
  lambda = 10^seq(-3, 1.5, length.out = 120)
)

# Train LASSO with glmnet via caret.
# center/scale is important for regularization-based models so the penalty
# treats predictors on comparable scales.
lasso_tuned <- train(
  x = train_data[, features],
  y = train_data[[outcome]],
  method = "glmnet",
  trControl = cv_control,
  tuneGrid = lasso_grid,
  preProcess = c("center", "scale")
)

# Extract coefficients at the CV-selected lambda.
coef_matrix <- as.matrix(coef(
  lasso_tuned$finalModel,
  s = lasso_tuned$bestTune$lambda
))

coef_table <- data.frame(
  term = rownames(coef_matrix),
  coefficient = as.numeric(coef_matrix[, 1]),
  is_interaction = rownames(coef_matrix) %in% interaction_features,
  stringsAsFactors = FALSE
)

interaction_coef_table <- coef_table[coef_table$is_interaction, ]

# Predict on the held-out test set using the best lambda selected by CV.
test_predictions <- predict(lasso_tuned, newdata = test_data[, features])

# Test-set RMSE and R-squared.
test_rmse <- sqrt(mean((test_data[[outcome]] - test_predictions)^2))
test_r2 <- 1 - sum((test_data[[outcome]] - test_predictions)^2) /
  sum((test_data[[outcome]] - mean(test_data[[outcome]]))^2)

# S-learner counterfactual prediction.
test_treated <- test_data
test_treated[[treatment]] <- 1
test_treated <- add_selected_interactions(test_treated)

test_control <- test_data
test_control[[treatment]] <- 0
test_control <- add_selected_interactions(test_control)

# Predict total wealth under each counterfactual scenario.
pred_treated <- predict(lasso_tuned, newdata = test_treated[, features])
pred_control <- predict(lasso_tuned, newdata = test_control[, features])

# Individual treatment effects and average treatment effect.
ite <- pred_treated - pred_control
ate <- mean(ite)

# Subgroup-level out-of-sample RMSE for diagnostic purposes.
treated_idx <- test_data$p401 == 1
control_idx <- test_data$p401 == 0

rmse_treated_subgroup <- sqrt(mean(
  (test_data$tw[treated_idx] - pred_treated[treated_idx])^2
))

rmse_control_subgroup <- sqrt(mean(
  (test_data$tw[control_idx] - pred_control[control_idx])^2
))

n_treated <- sum(treated_idx)
n_control <- sum(control_idx)

# Save artifacts for handoff to the analysis team.
saveRDS(lasso_tuned, "lasso_selected_model.rds")

test_results <- test_data
test_results$pred_treated <- pred_treated
test_results$pred_control <- pred_control
test_results$ite <- ite
test_results$row_id <- as.integer(rownames(test_data))
write.csv(test_results, "lasso_selected_test_results.csv", row.names = FALSE)

model_summary <- data.frame(
  metric = c("Test RMSE", "Test R-squared", "Average Treatment Effect"),
  value  = c(test_rmse, test_r2, ate)
)
write.csv(model_summary, "lasso_selected_model_summary.csv", row.names = FALSE)

write.csv(coef_table, "lasso_selected_coefficients.csv", row.names = FALSE)
write.csv(interaction_coef_table, "lasso_selected_interaction_coefficients.csv", row.names = FALSE)
