# ============================================================
# 401(k) Project Data Cleaning
# ============================================================
rm(list = ls())
library(tidyverse)
# Load raw data
df_raw <- read.csv("401k.csv")
df_raw <- read.csv("~/Desktop/401k.csv")
df_raw <- read.csv(file.choose())
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
summary(df_clean[, c("p401", "e401", "tw", "net_tfa", "net_n401",
"inc", "inc_k", "age", "educ", "fsize")])
write.csv(df_clean, "401k_clean.csv", row.names = FALSE)
file.exists("401k_clean.csv")
getwd()
cd ~/Desktop/Machine Learning
savehistory("401k_data_cleaning.R")
