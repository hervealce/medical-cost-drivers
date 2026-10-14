# Medical cost drivers - R version (dplyr + base stats)
suppressPackageStartupMessages(library(dplyr))
df <- read.csv("data/insurance.csv") |> distinct() |>
  mutate(is_smoker = smoker == "yes", obese = bmi >= 30)
cat("Rows after de-duplication:", nrow(df), "\n\n")

df |> group_by(is_smoker, obese) |>
  summarise(n = n(), avg = round(mean(charges)), .groups = "drop") |> print()

cat("\nOne-sided Welch t-test: obese > non-obese\n")
print(t.test(charges ~ factor(obese, levels = c(TRUE, FALSE)), data = df,
             alternative = "greater", var.equal = FALSE))
cat("Mann-Whitney (one-sided):\n")
print(wilcox.test(df$charges[df$obese], df$charges[!df$obese], alternative = "greater"))

fit <- lm(charges ~ age + bmi + children + is_smoker + sex + region + is_smoker:obese, data = df)
print(summary(fit))
