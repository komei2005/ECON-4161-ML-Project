# ============================================================
# 401(k) Project Full Code
# Data Cleaning + OLS/2SLS Replication + LASSO Model
# ============================================================

rm(list = ls())

# ============================================================
# 0. Libraries
# ============================================================

packages <- c("tidyverse", "ivreg", "caret", "glmnet", "scales")

installed <- rownames(installed.packages())

for (p in packages) {
  if (!(p %in% installed)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(ivreg)
  library(caret)
  library(glmnet)
  library(scales)
})

set.seed(42)

# ============================================================
# 1. Data Cleaning
# ============================================================

# Load raw data if available. If not, use cleaned data.
if (file.exists("401k.csv")) {
  df_raw <- read.csv("401k.csv")
} else if (file.exists("401k_clean.csv")) {
  df_raw <- read.csv("401k_clean.csv")
} else {
  stop("Neither 401k.csv nor 401k_clean.csv was found in the project folder.")
}

# Basic quality checks
cat("\nData dimensions:\n")
print(dim(df_raw))

cat("\nMissing values:\n")
print(sum(is.na(df_raw)))

cat("\nDuplicate rows:\n")
print(sum(duplicated(df_raw)))

cat("\n401(k) participation by eligibility:\n")
print(table(df_raw$p401, df_raw$e401))

cat("\nImpossible cases, participating but not eligible:\n")
print(sum(df_raw$p401 == 1 & df_raw$e401 == 0))

# Clean and create variables used in the project
df_clean <- df_raw %>%
  mutate(
    # Treatment and instrument
    p401 = as.integer(p401),
    e401 = as.integer(e401),

    # Binary controls
    db = as.integer(db),
    marr = as.integer(marr),
    male = as.integer(male),
    twoearn = as.integer(twoearn),
    pira = as.integer(pira),
    hown = as.integer(hown),

    # Income in thousands
    inc_k = inc / 1000,

    # Squared terms kept for possible future checks
    age_sq = age^2,
    inc_k_sq = inc_k^2,
    educ_sq = educ^2,
    fsize_sq = fsize^2,

    # Income groups matching the paper/project replication
    inc_group = cut(
      inc,
      breaks = c(-Inf, 10000, 20000, 30000, 40000, 50000, 75000, Inf),
      labels = c("<10k", "10-20k", "20-30k", "30-40k", "40-50k", "50-75k", ">75k"),
      right = FALSE
    ),

    # Age groups matching the paper/project replication
    age_group = cut(
      age,
      breaks = c(-Inf, 30, 36, 45, 55, Inf),
      labels = c("<30", "30-35", "36-44", "45-54", ">=55"),
      right = FALSE
    ),

    # Education groups
    educ_group = cut(
      educ,
      breaks = c(-Inf, 12, 13, 16, Inf),
      labels = c("<12", "12", "13-15", ">=16"),
      right = FALSE
    )
  )

# Save cleaned data
write.csv(df_clean, "401k_clean.csv", row.names = FALSE)

# Use cleaned data for all remaining analysis
data <- df_clean

# ============================================================
# 2. Summary Statistics for Report
# ============================================================

summary_by_participation <- data %>%
  group_by(p401) %>%
  summarise(
    n = n(),
    mean_total_wealth = mean(tw, na.rm = TRUE),
    median_total_wealth = median(tw, na.rm = TRUE),
    mean_net_financial_assets = mean(net_tfa, na.rm = TRUE),
    median_net_financial_assets = median(net_tfa, na.rm = TRUE),
    mean_income = mean(inc, na.rm = TRUE),
    mean_age = mean(age, na.rm = TRUE),
    share_homeowner = mean(hown, na.rm = TRUE),
    share_married = mean(marr, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(summary_by_participation, "summary_by_participation.csv", row.names = FALSE)

cat("\nSummary by 401(k) participation:\n")
print(summary_by_participation)

# ============================================================
# 3. OLS and 2SLS Replication
# ============================================================

outcomes <- c("net_tfa", "net_n401", "tw")
treatment <- "p401"
instrument <- "e401"

x_formula <- ~ inc_group + age_group + educ_group + marr +
  fsize + twoearn + db + pira + hown

controls_rhs <- as.character(x_formula)[2]

results <- data.frame(
  outcome = character(),
  model = character(),
  coef_p401 = numeric(),
  se_p401 = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

for (y in outcomes) {

  f_ols <- as.formula(
    paste0(y, " ~ ", treatment, " + ", controls_rhs)
  )

  f_iv <- as.formula(
    paste0(y, " ~ ", treatment, " + ", controls_rhs,
           " | ", instrument, " + ", controls_rhs)
  )

  m_ols <- lm(f_ols, data = data)
  m_iv <- ivreg(f_iv, data = data)

  ols_ctab <- summary(m_ols)$coefficients
  iv_ctab <- summary(m_iv, diagnostics = TRUE)$coefficients

  results <- rbind(
    results,
    data.frame(
      outcome = y,
      model = "OLS",
      coef_p401 = ols_ctab[treatment, "Estimate"],
      se_p401 = ols_ctab[treatment, "Std. Error"],
      p_value = ols_ctab[treatment, "Pr(>|t|)"]
    ),
    data.frame(
      outcome = y,
      model = "2SLS",
      coef_p401 = iv_ctab[treatment, "Estimate"],
      se_p401 = iv_ctab[treatment, "Std. Error"],
      p_value = iv_ctab[treatment, "Pr(>|t|)"]
    )
  )
}

# Clean replication table for report
replication_wide <- results %>%
  select(outcome, model, coef_p401) %>%
  mutate(
    outcome = case_when(
      outcome == "net_tfa" ~ "Net financial assets",
      outcome == "net_n401" ~ "Net non-401(k) assets",
      outcome == "tw" ~ "Total wealth",
      TRUE ~ outcome
    ),
    coef_p401 = round(coef_p401, 2)
  ) %>%
  pivot_wider(
    names_from = model,
    values_from = coef_p401
  )

write.csv(replication_wide, "replication_ols_2sls_wide.csv", row.names = FALSE)

cat("\nOLS/2SLS replication results:\n")
print(replication_wide)

# First stage
first_stage <- lm(
  as.formula(paste0(treatment, " ~ ", instrument, " + ", controls_rhs)),
  data = data
)

first_stage_result <- data.frame(
  variable = "e401",
  estimate = coef(summary(first_stage))["e401", "Estimate"],
  se = coef(summary(first_stage))["e401", "Std. Error"],
  p_value = coef(summary(first_stage))["e401", "Pr(>|t|)"]
)

write.csv(first_stage_result, "first_stage_result.csv", row.names = FALSE)

cat("\nFirst-stage result:\n")
print(first_stage_result)

# ============================================================
# 4. LASSO Model
# ============================================================

# Outcome: total wealth
outcome <- "tw"
treatment <- "p401"

# Predictor set for the LASSO wealth model.
# We exclude asset/wealth-component variables such as a401, tfa, net_tfa,
# tfa_he, hval, hmort, hequity, nifa, net_nifa, and net_n401 because they are
# mechanically related to total wealth and would create leakage.
# LASSO uses base predictors plus pairwise interactions, then shrinks less
# useful terms toward zero to reduce overfitting and improve interpretability.
predictors <- c(
  "age", "inc", "fsize", "educ", "db", "marr", "male",
  "twoearn", "pira", "hown"
)

# Add every pairwise interaction among treatment and predictors.
add_engineered_terms <- function(df, base_vars) {
  interaction_pairs <- combn(base_vars, 2, simplify = FALSE)
  for (pair in interaction_pairs) {
    interaction_name <- paste0(pair[1], "_x_", pair[2])
    df[[interaction_name]] <- df[[pair[1]]] * df[[pair[2]]]
  }
  df
}

base_vars <- c(treatment, predictors)
model_data <- add_engineered_terms(data[, c(outcome, base_vars)], base_vars)

# Stratified 70/30 train/test split by 401(k) participation.
train_index <- createDataPartition(model_data[[treatment]], p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

interaction_features <- setdiff(names(model_data), c(outcome, base_vars))
features <- c(base_vars, interaction_features)

# Tune LASSO penalty parameter using 10-fold cross-validation on the training set.
cv_control <- trainControl(method = "cv", number = 10)

lasso_grid <- expand.grid(
  alpha = 1,
  lambda = 10^seq(-3, 1.5, length.out = 120)
)

lasso_tuned <- train(
  x = train_data[, features],
  y = train_data[[outcome]],
  method = "glmnet",
  trControl = cv_control,
  tuneGrid = lasso_grid,
  preProcess = c("center", "scale")
)

# Coefficients at the CV-selected lambda.
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

write.csv(coef_table, "lasso_coefficients.csv", row.names = FALSE)

# Test-set prediction and performance.
test_predictions <- predict(lasso_tuned, newdata = test_data[, features])

test_rmse <- sqrt(mean((test_data[[outcome]] - test_predictions)^2))

test_r2 <- 1 - sum((test_data[[outcome]] - test_predictions)^2) /
  sum((test_data[[outcome]] - mean(test_data[[outcome]]))^2)

# S-learner counterfactual prediction:
# predict each test observation twice, once with p401 forced to 1 and once with
# p401 forced to 0. The difference is the model-predicted treatment effect.
# This should be interpreted as a predictive counterfactual estimate, not as a
# fully causal estimate, because 401(k) participation may be endogenous.
test_treated <- test_data
test_treated[[treatment]] <- 1
test_treated <- add_engineered_terms(test_treated, base_vars)

test_control <- test_data
test_control[[treatment]] <- 0
test_control <- add_engineered_terms(test_control, base_vars)

pred_treated <- predict(lasso_tuned, newdata = test_treated[, features])
pred_control <- predict(lasso_tuned, newdata = test_control[, features])

ite <- pred_treated - pred_control
ate <- mean(ite)

model_summary <- data.frame(
  metric = c("Test RMSE", "Test R-squared", "Average Predicted Treatment Effect"),
  value = c(test_rmse, test_r2, ate)
)

write.csv(model_summary, "model_summary.csv", row.names = FALSE)
write.csv(model_summary, "lasso_model_summary.csv", row.names = FALSE)

cat("\nLASSO model summary:\n")
print(model_summary)

# ============================================================
# 5. Heterogeneity by Income
# ============================================================

# Use fixed income bins that align with the replication categories.
test_income_group <- cut(
  test_data$inc,
  breaks = c(-Inf, 10000, 20000, 30000, 40000, 50000, 75000, Inf),
  labels = c("<10k", "10-20k", "20-30k", "30-40k", "40-50k", "50-75k", ">75k"),
  right = FALSE
)

heterogeneity_by_income <- data.frame(
  inc_group = test_income_group,
  ite = ite
) %>%
  group_by(inc_group) %>%
  summarise(
    n = n(),
    avg_treatment_effect = mean(ite, na.rm = TRUE),
    median_treatment_effect = median(ite, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(heterogeneity_by_income, "heterogeneity_by_income.csv", row.names = FALSE)
write.csv(
  heterogeneity_by_income %>% rename(income_group = inc_group, ate = avg_treatment_effect),
  "lasso_ate_by_income_group.csv",
  row.names = FALSE
)

cat("\nLASSO heterogeneity by income:\n")
print(heterogeneity_by_income)

# ============================================================
# 6. Graphs for Report
# ============================================================

if (!dir.exists("figures")) {
  dir.create("figures")
}

# Figure 1: Distribution of Total Wealth by 401(k) Participation
fig1 <- ggplot(data, aes(x = tw, fill = factor(p401))) +
  geom_density(alpha = 0.45) +
  scale_x_continuous(
    labels = dollar_format(),
    limits = quantile(data$tw, c(0.01, 0.99), na.rm = TRUE)
  ) +
  scale_fill_discrete(
    name = "401(k) Participation",
    labels = c("Non-participant", "Participant")
  ) +
  labs(
    title = "Distribution of Total Wealth by 401(k) Participation",
    x = "Total Wealth",
    y = "Density"
  ) +
  theme_minimal()

ggsave(
  filename = "figures/figure_1_total_wealth_density.png",
  plot = fig1,
  width = 8,
  height = 5
)

# Figure 2: Average Total Wealth by Income Group and Participation
fig2_data <- data %>%
  group_by(inc_group, p401) %>%
  summarise(
    mean_total_wealth = mean(tw, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    p401_label = ifelse(p401 == 1, "Participant", "Non-participant")
  )

fig2 <- ggplot(fig2_data, aes(x = inc_group, y = mean_total_wealth, fill = p401_label)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Average Total Wealth by Income Group and 401(k) Participation",
    x = "Income Group",
    y = "Average Total Wealth",
    fill = "401(k) Participation"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(
  filename = "figures/figure_2_avg_wealth_by_income_participation.png",
  plot = fig2,
  width = 9,
  height = 5
)

# Figure 3: Average LASSO Predicted Treatment Effect by Income Group
fig3 <- ggplot(heterogeneity_by_income, aes(x = inc_group, y = avg_treatment_effect)) +
  geom_col() +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Average Predicted Treatment Effect by Income Group",
    x = "Income Group",
    y = "Average LASSO Predicted Treatment Effect"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(
  filename = "figures/figure_3_predicted_effect_by_income.png",
  plot = fig3,
  width = 9,
  height = 5
)

# ============================================================
# 7. Finished
# ============================================================

cat("\n401k_full_code.R finished successfully.\n")
cat("Main outputs saved:\n")
cat("- 401k_clean.csv\n")
cat("- summary_by_participation.csv\n")
cat("- replication_ols_2sls_wide.csv\n")
cat("- first_stage_result.csv\n")
cat("- model_summary.csv\n")
cat("- lasso_model_summary.csv\n")
cat("- lasso_coefficients.csv\n")
cat("- heterogeneity_by_income.csv\n")
cat("- lasso_ate_by_income_group.csv\n")
cat("- figures/figure_1_total_wealth_density.png\n")
cat("- figures/figure_2_avg_wealth_by_income_participation.png\n")
cat("- figures/figure_3_predicted_effect_by_income.png\n")

# ============================================================
# Optional Random Forest Benchmark - Not Run in Final Script
# ============================================================
# This section is kept only as a record of the alternative model considered.
# The final report/code uses the LASSO model above because the group selected it
# as the preferred final model due to its interpretability and comparable predictive performance.
# To run this benchmark later, change if (FALSE) to if (TRUE) and make sure the
# ranger package is installed and loaded.

if (FALSE) {
  library(ranger)

  rf_predictors <- c(
    "age", "inc", "fsize", "educ", "db", "marr", "male",
    "twoearn", "pira", "hown"
  )

  rf_outcome <- "tw"
  rf_treatment <- "p401"
  rf_model_data <- data[, c(rf_outcome, rf_treatment, rf_predictors)]

  rf_train_index <- createDataPartition(
    rf_model_data[[rf_treatment]],
    p = 0.7,
    list = FALSE
  )

  rf_train_data <- rf_model_data[rf_train_index, ]
  rf_test_data <- rf_model_data[-rf_train_index, ]
  rf_features <- c(rf_treatment, rf_predictors)

  rf_cv_control <- trainControl(method = "cv", number = 10)

  rf_tune_grid <- expand.grid(
    mtry = c(2, 3, 4, 5, 6, 8, 10),
    splitrule = c("variance", "extratrees"),
    min.node.size = c(5, 10, 20)
  )

  rf_tuned <- train(
    x = rf_train_data[, rf_features],
    y = rf_train_data[[rf_outcome]],
    method = "ranger",
    trControl = rf_cv_control,
    tuneGrid = rf_tune_grid,
    num.trees = 500
  )

  rf_test_predictions <- predict(rf_tuned, newdata = rf_test_data[, rf_features])

  rf_test_rmse <- sqrt(mean((rf_test_data[[rf_outcome]] - rf_test_predictions)^2))

  rf_test_r2 <- 1 - sum((rf_test_data[[rf_outcome]] - rf_test_predictions)^2) /
    sum((rf_test_data[[rf_outcome]] - mean(rf_test_data[[rf_outcome]]))^2)

  rf_test_treated <- rf_test_data
  rf_test_treated[[rf_treatment]] <- 1

  rf_test_control <- rf_test_data
  rf_test_control[[rf_treatment]] <- 0

  rf_pred_treated <- predict(rf_tuned, newdata = rf_test_treated[, rf_features])
  rf_pred_control <- predict(rf_tuned, newdata = rf_test_control[, rf_features])

  rf_ite <- rf_pred_treated - rf_pred_control
  rf_ate <- mean(rf_ite)

  rf_model_summary <- data.frame(
    metric = c("RF Test RMSE", "RF Test R-squared", "RF Average Predicted Treatment Effect"),
    value = c(rf_test_rmse, rf_test_r2, rf_ate)
  )

  print(rf_model_summary)
}