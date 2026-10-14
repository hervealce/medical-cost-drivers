-- Medical cost drivers: cleaning + segment aggregates (DuckDB)
CREATE OR REPLACE TABLE policyholders AS
SELECT DISTINCT
    age, lower(trim(sex)) AS sex, bmi, children,
    lower(CAST(smoker AS VARCHAR)) IN ('yes','true') AS is_smoker,
    lower(trim(region)) AS region, charges,
    CASE WHEN bmi < 18.5 THEN '1 underweight'
         WHEN bmi < 25   THEN '2 healthy'
         WHEN bmi < 30   THEN '3 overweight'
         ELSE                 '4 obese' END AS bmi_band,
    CASE WHEN age < 30 THEN '18-29' WHEN age < 45 THEN '30-44'
         WHEN age < 60 THEN '45-59' ELSE '60+' END AS age_band
FROM read_csv_auto('data/insurance.csv');

-- Q1: rows after de-duplication
SELECT count(*) AS n_rows FROM policyholders;

-- Q2: smoker x BMI band
SELECT is_smoker, bmi_band, count(*) AS n,
       round(avg(charges), 0) AS avg_charges,
       round(median(charges), 0) AS median_charges
FROM policyholders GROUP BY ALL ORDER BY is_smoker, bmi_band;

-- Q3: region ranking with share of total spend (window function)
SELECT region, count(*) AS n, round(avg(charges), 0) AS avg_charges,
       round(100 * sum(charges) / sum(sum(charges)) OVER (), 1) AS pct_of_total_spend,
       rank() OVER (ORDER BY avg(charges) DESC) AS cost_rank
FROM policyholders GROUP BY region ORDER BY cost_rank;

-- Q4: cost ladder by age band, with smoker premium
SELECT age_band,
       round(avg(charges) FILTER (WHERE NOT is_smoker), 0) AS non_smoker_avg,
       round(avg(charges) FILTER (WHERE is_smoker), 0)     AS smoker_avg,
       round(avg(charges) FILTER (WHERE is_smoker) /
             avg(charges) FILTER (WHERE NOT is_smoker), 2) AS smoker_multiple
FROM policyholders GROUP BY age_band ORDER BY age_band;
