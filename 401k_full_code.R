# ============================================================
# 401(k) Project Full Code
# Data Cleaning + OLS/2SLS Replication + Random Forest Model
# ============================================================

rm(list = ls())

# ============================================================
# 0. Libraries
# ============================================================

packages <- c("tidyverse", "ivreg", "caret", "ranger", "scales")

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
  library(ranger)
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
# 4. Random Forest Model
# ============================================================


# Outcome: total wealth
outcome <- "tw"
treatment <- "p401"


# Predictor set for the Random Forest wealth model.
# We exclude asset/wealth-component variables such as a401, tfa, net_tfa,
# tfa_he, hval, hmort, hequity, nifa, net_nifa, and net_n401 because they are
# mechanically related to total wealth and would create leakage.
# We also exclude squared/grouped variables because Random Forests handle
# nonlinearities natively, while those grouped variables are used for the
# replication and heterogeneity analysis.

predictors <- c(
  "age", "inc", "fsize", "educ", "db", "marr", "male",
  "twoearn", "pira", "hown"
)

model_data <- data[, c(outcome, treatment, predictors)]


# Stratified 70/30 train/test split by 401(k) participation.
# Stratification keeps the participant/nonparticipant mix similar in both samples.
# The test set is held out for final predictive evaluation and treatment-effect estimation.

train_index <- createDataPartition(model_data[[treatment]], p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

features <- c(treatment, predictors)

# Cross-validation and tuning
# Tune Random Forest hyperparameters using 10-fold cross-validation on the training set.
# The held-out test set is not used during tuning.

cv_control <- trainControl(method = "cv", number = 10)

tune_grid <- expand.grid(
  mtry = c(2, 3, 4, 5, 6, 8, 10),
  splitrule = c("variance", "extratrees"),
  min.node.size = c(5, 10, 20)
)

rf_tuned <- train(
  x = train_data[, features],
  y = train_data[[outcome]],
  method = "ranger",
  trControl = cv_control,
  tuneGrid = tune_grid,
  num.trees = 500
)

# Test-set prediction
test_predictions <- predict(rf_tuned, newdata = test_data[, features])

test_rmse <- sqrt(mean((test_data[[outcome]] - test_predictions)^2))

test_r2 <- 1 - sum((test_data[[outcome]] - test_predictions)^2) /
  sum((test_data[[outcome]] - mean(test_data[[outcome]]))^2)

# Counterfactual prediction exercise
# S-learner counterfactual prediction:
# predict each test observation twice, once with p401 forced to 1 and once with
# p401 forced to 0. The difference is the model-predicted treatment effect.
# This should be interpreted as a predictive counterfactual estimate, not as a
# fully causal estimate, because 401(k) participation may be endogenous.
test_treated <- test_data
test_treated[[treatment]] <- 1

test_control <- test_data
test_control[[treatment]] <- 0

pred_treated <- predict(rf_tuned, newdata = test_treated[, features])
pred_control <- predict(rf_tuned, newdata = test_control[, features])

ite <- pred_treated - pred_control
ate <- mean(ite)

model_summary <- data.frame(
  metric = c("Test RMSE", "Test R-squared", "Average Predicted Treatment Effect"),
  value = c(test_rmse, test_r2, ate)
)

write.csv(model_summary, "model_summary.csv", row.names = FALSE)

cat("\nRandom Forest model summary:\n")
print(model_summary)

# ============================================================
# 5. Heterogeneity by Income
# ============================================================

test_results <- test_data
test_results$pred_treated <- pred_treated
test_results$pred_control <- pred_control
test_results$ite <- ite
test_results$row_id <- as.integer(rownames(test_data))

data_with_row_id <- data %>%
  mutate(row_id = row_number()) %>%
  select(row_id, inc_group)

test_results_with_groups <- test_results %>%
  left_join(data_with_row_id, by = "row_id")

heterogeneity_by_income <- test_results_with_groups %>%
  group_by(inc_group) %>%
  summarise(
    n = n(),
    avg_treatment_effect = mean(ite, na.rm = TRUE),
    median_treatment_effect = median(ite, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(heterogeneity_by_income, "heterogeneity_by_income.csv", row.names = FALSE)

cat("\nHeterogeneity by income:\n")
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

# Figure 3: Average Predicted Treatment Effect by Income Group
fig3 <- ggplot(heterogeneity_by_income, aes(x = inc_group, y = avg_treatment_effect)) +
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
cat("- heterogeneity_by_income.csv\n")
cat("- figures/figure_1_total_wealth_density.png\n")
cat("- figures/figure_2_avg_wealth_by_income_participation.png\n")
cat("- figures/figure_3_predicted_effect_by_income.png\n")