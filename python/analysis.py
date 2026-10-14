"""Medical cost drivers: SQL aggregates, hypothesis tests, explanatory model."""
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from style import style, BROWN, ACCENT, CREAM

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; style()
con = duckdb.connect()
import os; os.chdir(ROOT)
sql = (ROOT / "sql/01_cost_drivers.sql").read_text()
stmts = [s for s in sql.split(";") if s.strip()]
con.execute(stmts[0])
for s in stmts[1:]:
    print(con.execute(s).df().to_string(index=False), "\n")
df = con.execute("SELECT * FROM policyholders ORDER BY age, sex, bmi, children, region, charges").df()
res = {"n_rows": len(df)}

# --- Hypothesis 1: obese (BMI>=30) vs not, one-sided Welch + Mann-Whitney
hi, lo = df.loc[df.bmi >= 30, "charges"], df.loc[df.bmi < 30, "charges"]
t = stats.ttest_ind(hi, lo, equal_var=False, alternative="greater")
u = stats.mannwhitneyu(hi, lo, alternative="greater")
d = (hi.mean() - lo.mean()) / np.sqrt((hi.var() + lo.var()) / 2)
res["bmi_test"] = dict(mean_obese=hi.mean(), mean_other=lo.mean(), welch_p=t.pvalue,
                       mwu_p=u.pvalue, cohens_d=d)
# --- Hypothesis 2: smokers vs non-smokers
s1, s0 = df.loc[df.is_smoker, "charges"], df.loc[~df.is_smoker, "charges"]
t2 = stats.ttest_ind(s1, s0, equal_var=False)
res["smoker_test"] = dict(mean_smoker=s1.mean(), mean_non=s0.mean(), welch_p=t2.pvalue,
                          cohens_d=(s1.mean()-s0.mean())/np.sqrt((s1.var()+s0.var())/2))
# --- Interaction: does BMI matter differently for smokers?
inter = df.groupby(["is_smoker", df.bmi >= 30])["charges"].mean().unstack()
res["interaction_means"] = {f"smoker={k}": {f"obese={c}": round(v, 0) for c, v in r.items()} for k, r in inter.iterrows()}

# --- Explanatory models
X = pd.get_dummies(df[["age", "bmi", "children", "is_smoker", "sex", "region"]], drop_first=True).astype(float)
X["smoker_x_obese"] = X["is_smoker"] * (df.bmi >= 30)
y = df["charges"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
for name, m in {"linear_regression": LinearRegression(),
                "gradient_boosting": GradientBoostingRegressor(random_state=42, max_depth=3)}.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    cv = cross_val_score(m, X, y, cv=5, scoring="r2")
    res[name] = dict(test_r2=r2_score(yte, p), test_mae=mean_absolute_error(yte, p), cv_r2_mean=cv.mean())
lr = LinearRegression().fit(X, y)
res["lr_coefficients"] = dict(zip(X.columns, lr.coef_.round(0)))

# --- Figures
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df, x="charges", hue=df.is_smoker.map({True: "Smoker", False: "Non-smoker"}),
             bins=40, palette=[ACCENT, BROWN], alpha=.75, ax=ax)
ax.set(title="Annual medical charges: smokers vs non-smokers", xlabel="Charges ($)", ylabel="Policyholders")
ax.get_legend().set_title(""); fig.tight_layout(); fig.savefig(FIG / "charges_by_smoker.png"); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for flag, col, lab in [(False, BROWN, "Non-smoker"), (True, ACCENT, "Smoker")]:
    s = df[df.is_smoker == flag]; ax.scatter(s.bmi, s.charges, s=14, alpha=.6, color=col, label=lab)
ax.axvline(30, ls="--", color="#888"); ax.text(30.3, df.charges.max()*.95, "BMI 30", color="#666")
ax.set(title="BMI only drives cost for smokers", xlabel="BMI", ylabel="Charges ($)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "bmi_vs_charges.png"); plt.close()

seg = df.groupby(["age_band", "is_smoker"])["charges"].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 5))
seg.plot.bar(ax=ax, color=[BROWN, ACCENT], rot=0)
ax.set(title="Average charges by age band", xlabel="Age band", ylabel="Avg charges ($)")
ax.legend(["Non-smoker", "Smoker"]); fig.tight_layout(); fig.savefig(FIG / "age_band_costs.png"); plt.close()

out = json.loads(json.dumps(res, default=float))
(ROOT / "python/results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
