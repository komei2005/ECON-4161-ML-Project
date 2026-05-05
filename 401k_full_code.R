# ============================================================
# 401(k) Project Full Code
# Data Cleaning + OLS/2SLS Replication + Random Forest Model
# ============================================================

rm(list = ls())

# ============================================================
# 0. Libraries
# ============================================================

packages <- c("tidyverse", "ivreg", "caret", "ranger")

installed <- rownames(installed.packages())

for (p in packages) {
  if (!(p %in% installed)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}

library(tidyverse)
library(ivreg)
library(caret)
library(ranger)

set.seed(42)

# ============================================================
# 1. Data Cleaning
# ============================================================

# Load raw data if available.
# If 401k.csv is not in the folder, use existing 401k_clean.csv.
if (file.exists("401k.csv")) {
  df_raw <- read.csv("401k.csv")
} else if (file.exists("401k_clean.csv")) {
  df_raw <- read.csv("401k_clean.csv")
} else {
  stop("Neither 401k.csv nor 401k_clean.csv was found in the project folder.")
}

# Check dimensions
dim(df_raw)

# Check column names
names(df_raw)

# Check variable types
str(df_raw)

# Check for missing values in each column
colSums(is.na(df_raw))

# Check duplicate rows
sum(duplicated(df_raw))

# 401(k) participation by eligibility
table(df_raw$p401, df_raw$e401)

# Impossible cases: participating but not eligible
sum(df_raw$p401 == 1 & df_raw$e401 == 0)

# Participants with no 401(k) assets
sum(df_raw$p401 == 1 & df_raw$a401 <= 0)

# Non-participants with positive 401(k) assets
sum(df_raw$p401 == 0 & df_raw$a401 > 0)

# Age range
range(df_raw$age)

# Income range
range(df_raw$inc)

# Wealth variable ranges
range(df_raw$tw)
range(df_raw$net_tfa)
range(df_raw$net_n401)

df_clean <- df_raw %>%
  mutate(
    # Treatment / eligibility variables
    p401 = as.integer(p401),
    e401 = as.integer(e401),

    # Binary controls
    db = as.integer(db),
    marr = as.integer(marr),
    male = as.integer(male),
    twoearn = as.integer(twoearn),
    pira = as.integer(pira),
    hown = as.integer(hown),

    # Income in thousands for easier interpretation later
    inc_k = inc / 1000,

    # Squared terms for later flexible models
    age_sq = age^2,
    inc_k_sq = inc_k^2,
    educ_sq = educ^2,
    fsize_sq = fsize^2,

    # Education group
    educ_group = case_when(
      educ < 12 ~ "Less than high school",
      educ == 12 ~ "High school",
      educ > 12 & educ < 16 ~ "Some college",
      educ >= 16 ~ "College or more",
      TRUE ~ "Unknown"
    ),

    # Age group
    age_group = case_when(
      age >= 25 & age <= 34 ~ "25-34",
      age >= 35 & age <= 44 ~ "35-44",
      age >= 45 & age <= 54 ~ "45-54",
      age >= 55 & age <= 64 ~ "55-64",
      TRUE ~ "Unknown"
    ),

    # Income group based on quartiles
    inc_group = case_when(
      inc <= quantile(inc, 0.25, na.rm = TRUE) ~ "Lowest income quartile",
      inc <= quantile(inc, 0.50, na.rm = TRUE) ~ "Second income quartile",
      inc <= quantile(inc, 0.75, na.rm = TRUE) ~ "Third income quartile",
      TRUE ~ "Highest income quartile"
    )
  )

# Check cleaned data
dim(df_clean)
names(df_clean)

summary(df_clean[, c(
  "p401", "e401", "tw", "net_tfa", "net_n401",
  "inc", "inc_k", "age", "educ", "fsize"
)])

# Save cleaned data
write.csv(df_clean, "401k_clean.csv", row.names = FALSE)

file.exists("401k_clean.csv")
getwd()

# ============================================================
# 2. OLS and 2SLS Replication
# ============================================================

# load the cleaned data
data <- read.csv("401k_clean.csv")

# outcome variables
# The outcomes Y are the three previously mentioned measures of wealth
# [total wealth, net financial assets, and net non-401(k) financial assets]
# Keep this order for all output tables.
outcomes <- c("net_tfa", "net_n401", "tw")

# treatment variable + instrument
treatment <- "p401"
instrument <- "e401"

# X consists of dummies for income category, dummies for age category,
# dummies for education category, a marital status indicator, family size,
# two-earner status, DB pension status, IRA participation status, homeownership
# status, and a constant.

# dummies for inc, age, and educ
data$inc_group <- cut(
  data$inc,
  breaks = c(-Inf, 10000, 20000, 30000, 40000, 50000, 75000, Inf),
  labels = c("<10k", "10-20k", "20-30k", "30-40k", "40-50k", "50-75k", ">75k"),
  right = FALSE
)

data$age_group <- cut(
  data$age,
  breaks = c(-Inf, 30, 36, 45, 55, Inf),
  labels = c("<30", "30-35", "36-44", "45-54", ">=55"),
  right = FALSE
)

data$educ_group <- cut(
  data$educ,
  breaks = c(-Inf, 12, 13, 16, Inf),
  labels = c("<12", "12", "13-15", ">=16"),
  right = FALSE
)

# non dummy covariates
base_covariates <- c("marr", "fsize", "twoearn", "db", "pira", "hown")

# formula including dummies + controls
x_formula <- ~ inc_group + age_group + educ_group + marr +
  fsize + twoearn + db + pira + hown

# shared controls for OLS and 2SLS
controls_rhs <- as.character(x_formula)[2]

# storage
ols_models <- list()
iv_models <- list()

results <- data.frame(
  outcome = character(),
  model = character(),
  coef_p401 = numeric(),
  se_p401 = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

for (y in outcomes) {
  # Formulas
  f_ols <- as.formula(
    paste0(y, " ~ ", treatment, " + ", controls_rhs)
  )

  f_iv <- as.formula(
    paste0(y, " ~ ", treatment, " + ", controls_rhs,
           " | ", instrument, " + ", controls_rhs)
  )

  # fit models
  m_ols <- lm(f_ols, data = data)
  m_iv  <- ivreg(f_iv, data = data)

  ols_models[[y]] <- m_ols
  iv_models[[y]] <- m_iv

  # Extract p401 rows
  ols_ctab <- summary(m_ols)$coefficients
  iv_ctab <- summary(m_iv, diagnostics = TRUE)$coefficients

  # Append results
  results <- rbind(
    results,
    data.frame(
      outcome = y,
      model = "OLS",
      coef_p401 = ols_ctab[treatment, "Estimate"],
      se_p401   = ols_ctab[treatment, "Std. Error"],
      p_value   = ols_ctab[treatment, "Pr(>|t|)"],
      stringsAsFactors = FALSE
    ),
    data.frame(
      outcome = y,
      model = "2SLS",
      coef_p401 = iv_ctab[treatment, "Estimate"],
      se_p401   = iv_ctab[treatment, "Std. Error"],
      p_value   = iv_ctab[treatment, "Pr(>|t|)"],
      stringsAsFactors = FALSE
    )
  )
}

# display results in order of report
results_display <- results[, c("outcome", "model", "coef_p401")]
print(results_display)

write.csv(results, "replication_ols_2sls_results.csv", row.names = FALSE)
write.csv(results_display, "replication_ols_2sls_display.csv", row.names = FALSE)

# wide version for report
results_wide <- results_display %>%
  pivot_wider(
    names_from = model,
    values_from = coef_p401
  )

print(results_wide)
write.csv(results_wide, "replication_ols_2sls_wide.csv", row.names = FALSE)

# first stage
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

print(first_stage_result)
write.csv(first_stage_result, "first_stage_result.csv", row.names = FALSE)

# ============================================================
# 3. Random Forest Model
# ============================================================

# 401(k) Treatment Effect -> Random Forest Model
# Komei, Kai, Sheriloye, Toye

# this is the cleaned dataset we now have from our data cleaning process
data <- read.csv("401k_clean.csv")

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

print(rf_tuned)
print(test_rmse)
print(test_r2)

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

# Average treatment effect from the ML counterfactual prediction exercise.
# This is reported in dollars and compared to OLS and 2SLS coefficients.
# Interpret cautiously because 401(k) participation may be endogenous.
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

print(rmse_treated_subgroup)
print(rmse_control_subgroup)
print(n_treated)
print(n_control)

# Save artifacts for handoff to the analysis team.
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

print(model_summary)

# ============================================================
# 4. Extra Outputs for Report Tables
# ============================================================

# Summary statistics by 401(k) participation
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

print(summary_by_participation)
write.csv(summary_by_participation, "summary_by_participation.csv", row.names = FALSE)

# Heterogeneity by income groups using test_results
data_with_row_id <- data %>%
  mutate(row_id = row_number()) %>%
  select(row_id, inc, educ, age, inc_group, age_group, educ_group)

test_results_with_groups <- test_results %>%
  left_join(data_with_row_id, by = "row_id")

heterogeneity_income <- test_results_with_groups %>%
  group_by(inc_group) %>%
  summarise(
    n = n(),
    avg_treatment_effect = mean(ite, na.rm = TRUE),
    median_treatment_effect = median(ite, na.rm = TRUE),
    .groups = "drop"
  )

print(heterogeneity_income)
write.csv(heterogeneity_income, "heterogeneity_by_income.csv", row.names = FALSE)

heterogeneity_education <- test_results_with_groups %>%
  group_by(educ_group) %>%
  summarise(
    n = n(),
    avg_treatment_effect = mean(ite, na.rm = TRUE),
    median_treatment_effect = median(ite, na.rm = TRUE),
    .groups = "drop"
  )

print(heterogeneity_education)
write.csv(heterogeneity_education, "heterogeneity_by_education.csv", row.names = FALSE)

# ============================================================
# Graphs for Report
# ============================================================

# install/load scales for formatting
if (!("scales" %in% rownames(installed.packages()))) {
  install.packages("scales", repos = "https://cloud.r-project.org")
}
library(scales)

# make figures folder if it doesn't exist
if (!dir.exists("figures")) {
  dir.create("figures")
}

# ------------------------------------------------
# Figure 1: Distribution of Total Wealth by 401(k) Participation
# ------------------------------------------------

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

# ------------------------------------------------
# Figure 2: Average Total Wealth by Income Group and Participation
# ------------------------------------------------

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

# ------------------------------------------------
# Figure 3: Average Predicted Treatment Effect by Income Group
# ------------------------------------------------

fig3 <- ggplot(heterogeneity_income, aes(x = inc_group, y = avg_treatment_effect)) +
  geom_col() +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Average Predicted Treatment Effect by Income Group",
    x = "Income Group",
    y = "Average Predicted Treatment Effect"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(
  filename = "figures/figure_3_predicted_effect_by_income.png",
  plot = fig3,
  width = 9,
  height = 5
)

cat("\nSaved 3 main report graphs in the figures folder.\n")

# ============================================================
# 5. Finished
# ============================================================

cat("\n401k_full_code.R finished successfully.\n")
cat("Main outputs saved:\n")
cat("- 401k_clean.csv\n")
cat("- replication_ols_2sls_results.csv\n")
cat("- replication_ols_2sls_display.csv\n")
cat("- replication_ols_2sls_wide.csv\n")
cat("- first_stage_result.csv\n")
cat("- rf_model.rds\n")
cat("- test_results.csv\n")
cat("- model_summary.csv\n")
cat("- summary_by_participation.csv\n")
cat("- heterogeneity_by_income.csv\n")
cat("- heterogeneity_by_education.csv\n")
cat("- figures/figure_1_total_wealth_density.png\n")
cat("- figures/figure_2_avg_wealth_by_income_participation.png\n")
cat("- figures/figure_3_predicted_effect_by_income.png\n")