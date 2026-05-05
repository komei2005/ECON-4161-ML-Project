library(ivreg)

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


# Run OLS and 2SLS for each outcome

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