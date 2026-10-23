# %% [markdown]
# # Experiment 13: Implementing Responsible AI and Bias Mitigation Techniques
#
# **Aim:** Detect, measure and mitigate bias in a machine-learning model, and apply responsible-AI practices that also
# matter for generative AI: fairness metrics, mitigation (pre-, in- and post-processing), explainability, privacy and
# output safety filtering, documented in a model card.

# %% title: Imports and setup
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
                               demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
rng = np.random.default_rng(13)
print("setup complete")

# %% [markdown]
# ## 1. A dataset with historical bias
# We simulate a hiring data set. The true qualification of candidates is identically distributed in both groups, but the
# *historical decisions* (the labels) penalise group 1 (e.g. women), and a "career_gap" feature acts as a **proxy** for the
# protected attribute. This mirrors how real data encodes past discrimination.

# %% title: Creating and inspecting the biased data set
n = 8000
sex = rng.binomial(1, 0.45, n)                                  # protected attribute: 0 = group A, 1 = group B
experience = np.clip(rng.normal(6, 3, n), 0, None)
education = rng.normal(0, 1, n)
skill_test = rng.normal(0, 1, n)
career_gap = np.clip(rng.normal(0.3 + 1.6 * sex, 0.8, n), 0, None)   # proxy: strongly correlated with group
merit = 0.25 * experience + 0.9 * education + 1.1 * skill_test          # true qualification (same in both groups)
logit = (merit - merit.mean()) / merit.std() * 1.6 - 1.0 * sex - 0.4 * career_gap + 0.2   # biased historical decision
hired = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame(dict(experience=experience, education=education, skill_test=skill_test, career_gap=career_gap, sex=sex, hired=hired))
print(df.head(), "\n")
print("group sizes:", df.sex.value_counts().to_dict())
print("mean merit by group   :", df.assign(m=merit).groupby("sex").m.mean().round(3).to_dict(), "(equal by construction)")
print("historical hire rate  :", df.groupby("sex").hired.mean().round(3).to_dict())
print("correlation(career_gap, sex) = %.2f" % df.career_gap.corr(df.sex))
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
ax[0].bar(["group A", "group B"], df.groupby("sex").hired.mean(), color=["tab:blue", "tab:orange"]); ax[0].set_title("Historical hiring rate")
for g, c in ((0, "tab:blue"), (1, "tab:orange")):
    ax[1].hist(df[df.sex == g].career_gap, bins=25, alpha=.6, color=c, label=f"group {'AB'[g]}")
    ax[2].hist(merit[sex == g], bins=25, alpha=.6, color=c, label=f"group {'AB'[g]}")
ax[1].set_title("Proxy feature: career_gap"); ax[1].legend(); ax[2].set_title("True merit (same distribution)"); ax[2].legend()
plt.tight_layout(); plt.show()

# %% title: Train/test split
features = ["experience", "education", "skill_test", "career_gap"]
Xtr, Xte, ytr, yte, str_, ste = train_test_split(df[features + ["sex"]], df.hired, df.sex, test_size=0.3, random_state=0, stratify=df.hired)
print("train:", Xtr.shape, "test:", Xte.shape)

# %% [markdown]
# ## 2. Measuring bias in a baseline model

# %% title: Baseline model and fairness metrics
def report(y_true, y_pred, sens, name):
    mf = MetricFrame(metrics={"accuracy": accuracy_score, "selection_rate": selection_rate, "TPR": true_positive_rate, "FPR": false_positive_rate},
                     y_true=y_true, y_pred=y_pred, sensitive_features=sens)
    dp = demographic_parity_difference(y_true, y_pred, sensitive_features=sens)
    di = demographic_parity_ratio(y_true, y_pred, sensitive_features=sens)
    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=sens)
    tpr_gap = mf.by_group["TPR"].max() - mf.by_group["TPR"].min()
    return dict(model=name, accuracy=accuracy_score(y_true, y_pred), dp_diff=dp, disparate_impact=di, eq_odds_diff=eo, tpr_gap=tpr_gap), mf
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr[features + ["sex"]], ytr)
res, mf = [], None
r, mf = report(yte, base.predict(Xte[features + ["sex"]]), ste, "baseline (uses sex)"); res.append(r)
print("Per-group metrics of the baseline model:\n", mf.by_group.round(3), "\n")
print("Demographic-parity difference : %.3f (0 = fair)" % r["dp_diff"])
print("Disparate-impact ratio        : %.3f (4/5 rule requires >= 0.80)" % r["disparate_impact"])
print("Equalised-odds difference     : %.3f" % r["eq_odds_diff"])

# %% [markdown]
# ## 3. Bias mitigation
# 1. **Fairness through unawareness** - drop the protected attribute (does it help when proxies exist?).
# 2. **Pre-processing: reweighing** - give each (group, label) cell the weight $w=\frac{P(g)P(y)}{P(g,y)}$ so that group and label become statistically independent.
# 3. **In-processing: constrained optimisation** - `ExponentiatedGradient` with a demographic-parity / equalised-odds constraint.
# 4. **Post-processing: group-specific thresholds** - `ThresholdOptimizer` adjusts decision thresholds per group.

# %% title: Applying the four mitigation strategies
Xf_tr, Xf_te = Xtr[features], Xte[features]
scaler = StandardScaler().fit(Xf_tr); Xs_tr, Xs_te = scaler.transform(Xf_tr), scaler.transform(Xf_te)   # ExponentiatedGradient needs a plain estimator
# 1. unawareness
m1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xf_tr, ytr)
r, _ = report(yte, m1.predict(Xf_te), ste, "1 unawareness (drop sex)"); res.append(r)
# 2. reweighing
w = np.ones(len(ytr))
for g in (0, 1):
    for y in (0, 1):
        cell = (str_.values == g) & (ytr.values == y)
        w[cell] = (str_.values == g).mean() * (ytr.values == y).mean() / cell.mean()
print("reweighing weights:", {f"group {'AB'[g]}, hired={y}": round(float(w[(str_.values == g) & (ytr.values == y)][0]), 3) for g in (0, 1) for y in (0, 1)})
m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
m2.fit(Xtr[features + ["sex"]], ytr, logisticregression__sample_weight=w)
r, _ = report(yte, m2.predict(Xte[features + ["sex"]]), ste, "2 reweighing (pre)"); res.append(r)
# 3. in-processing
for nm, cons in (("3a exp. gradient: DP", DemographicParity()), ("3b exp. gradient: EO", EqualizedOdds())):
    eg = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=cons, eps=0.02)
    eg.fit(Xs_tr, ytr, sensitive_features=str_)
    r, _ = report(yte, eg.predict(Xs_te), ste, nm + " (in)"); res.append(r)
# 4. post-processing
for nm, cons in (("4a threshold opt: DP", "demographic_parity"), ("4b threshold opt: EO", "equalized_odds")):
    to = ThresholdOptimizer(estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xf_tr, ytr),
                            constraints=cons, prefit=True, predict_method="predict_proba")
    to.fit(Xf_tr, ytr, sensitive_features=str_)
    r, _ = report(yte, to.predict(Xf_te, sensitive_features=ste, random_state=0), ste, nm + " (post)"); res.append(r)
table = pd.DataFrame(res).set_index("model").round(3)
print("\n" + table.to_string())

# %% title: Accuracy-fairness trade-off of all models
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
t = table.reset_index()
cols = plt.cm.tab10(np.arange(len(t)))
ax[0].barh(t.model, t.dp_diff, color=cols); ax[0].invert_yaxis(); ax[0].set_title("Demographic-parity difference (lower = fairer)", fontsize=9)
ax[0].tick_params(axis="y", labelsize=7)
for i, row in t.iterrows(): ax[1].scatter(row.dp_diff, row.accuracy, s=70, color=cols[i], label=row.model)
ax[1].set(xlabel="demographic-parity difference", ylabel="accuracy", title="Trade-off between fairness and accuracy"); ax[1].legend(fontsize=6, loc="best")
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.bar(t.model, t.disparate_impact, color=cols); ax.axhline(0.8, c="r", ls="--", label="0.8 threshold (4/5 rule)")
ax.set_ylim(0, 1.05); ax.set_title("Disparate-impact ratio"); ax.legend(); plt.xticks(rotation=30, ha="right", fontsize=7); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Explainability, privacy, safety and documentation

# %% title: Explainability - permutation importance exposes the proxy feature
gb = GradientBoostingClassifier(random_state=0).fit(Xf_tr, ytr)
pi = permutation_importance(gb, Xf_te, yte, n_repeats=8, random_state=0)
imp = pd.Series(pi.importances_mean, index=features).sort_values()
print("Gradient-boosting model WITHOUT the sex column - permutation importance (drop in accuracy):")
print(imp.round(4).to_string())
rep = report(yte, gb.predict(Xf_te), ste, "gb")[0]
print("\nEven without 'sex', DP difference of the boosted model = %.3f -> bias flows in through the proxy 'career_gap'." % rep["dp_diff"])
fig, ax = plt.subplots(figsize=(6, 2.6)); imp.plot.barh(ax=ax, color="tab:purple"); ax.set_title("Permutation importance"); plt.tight_layout(); plt.show()

# %% title: Privacy - the Laplace mechanism for differentially private statistics
def dp_rate(group_mask, eps):
    cnt = int((ytr.values[group_mask] == 1).sum()); size = int(group_mask.sum())
    noisy_cnt = cnt + rng.laplace(0, 1 / eps)            # sensitivity of a count = 1
    return noisy_cnt / size
true_rate = ytr.values[str_.values == 1].mean()
print("True hiring rate of group B in the training data: %.4f" % true_rate)
fig, ax = plt.subplots(figsize=(6, 2.8))
for eps in (0.01, 0.1, 1.0):
    est = [dp_rate(str_.values == 1, eps) for _ in range(500)]
    err = np.mean(np.abs(np.array(est) - true_rate))
    print(f"epsilon = {eps:<5}: mean absolute error of the private estimate = {err:.4f}")
    ax.hist(est, bins=30, alpha=.55, label=f"eps={eps}")
ax.axvline(true_rate, c="k", ls="--"); ax.set_title("Differentially private estimate of a rate"); ax.legend(); plt.tight_layout(); plt.show()

# %% title: Safety filter for generated text (PII, toxicity block-list, bias terms)
BLOCK = {"idiot", "stupid", "hate", "kill"}
PATTERNS = {"EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+", "PHONE": r"\b(?:\+?\d{1,3}[ -]?)?\d{10}\b", "AADHAAR-LIKE": r"\b\d{4} \d{4} \d{4}\b"}
def safety_filter(text):
    flags = []
    for name, pat in PATTERNS.items():
        if re.search(pat, text): flags.append(name); text = re.sub(pat, f"[{name} REMOVED]", text)
    hits = [w for w in re.findall(r"\w+", text.lower()) if w in BLOCK]
    if hits: flags.append("TOXICITY:" + ",".join(sorted(set(hits)))); text = "[BLOCKED: unsafe content]"
    return text, flags
samples = ["Contact the author at jane.doe@example.com or 9876543210 for the full report.",
           "You are an idiot and I hate this.", "My id number is 1234 5678 9012.", "The quick brown fox jumps over the lazy dog."]
for s in samples:
    out, flags = safety_filter(s)
    print(f"INPUT : {s}\nOUTPUT: {out}\nFLAGS : {flags or 'none'}\n")

# %% title: Model card summarising the responsible-AI evaluation
base_r, best = table.iloc[0], table["dp_diff"].idxmin()
card = f"""MODEL CARD - Hiring-score classifier (educational example)
Intended use     : demonstrate fairness auditing; NOT for real hiring decisions.
Training data    : 8,000 synthetic candidates, groups A/B (55%/45%), labels contain simulated historical bias.
Model            : logistic regression, 4 numeric features (+ protected attribute in baseline).
Baseline results : accuracy {base_r.accuracy:.3f}, DP difference {base_r.dp_diff:.3f}, disparate impact {base_r.disparate_impact:.3f}.
Best fairness    : '{best}' -> DP difference {table.loc[best, 'dp_diff']:.3f}, accuracy {table.loc[best, 'accuracy']:.3f}.
Known limitations: proxy feature (career_gap) leaks group membership; fairness metrics conflict with each other;
                   synthetic data cannot capture real-world bias.
Mitigations      : reweighing, constrained optimisation, threshold optimisation; PII/toxicity output filter.
Monitoring       : re-audit selection rates per group on every model update."""
print(card)
