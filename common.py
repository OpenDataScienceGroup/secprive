"""Shared loading, recoding and analysis functions for group M."""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

DATA = "/home/claude/proj/data/"
REF = pd.Timestamp("2025-07-01")  # date of data collection
RNG_SEED = 2025

EDU_GROUP = {
    "Primary education": "Primary",
    "Upper secondary education": "Secondary/VET",
    "Vocational Education and Training (VET)": "Secondary/VET",
    "Short cycle higher education": "Short/medium higher",
    "Vocational bachelors educations": "Short/medium higher",
    "Bachelors programmes": "Short/medium higher",
    "Masters programmes": "Long higher",
    "PhD programmes": "Long higher",
    "Not stated": "Not stated",
}
EDU_ORDER = ["Primary", "Secondary/VET", "Short/medium higher", "Long higher"]
AGE_BINS = [0, 35, 50, 65, 200]
AGE_LABELS = ["20-34", "35-49", "50-64", "65+"]


def age_at_ref(dob):
    a = REF.year - dob.dt.year
    a -= ((dob.dt.month > REF.month) | ((dob.dt.month == REF.month) & (dob.dt.day > REF.day))).astype(int)
    return a


def load_private():
    return pd.read_excel(DATA + "private_dataM.xlsx")


def load_register():
    return pd.read_excel(DATA + "public_data_registerM.xlsx")


def load_results():
    d = pd.read_excel(DATA + "public_data_resultsM.xlsx")
    d = d.rename(columns={d.columns[0]: "unit", "Invalid ballots": "Invalid"})
    return d.set_index("unit")


def standardise_raw(p):
    """Bring raw data into the analysis schema used for both raw and anonymised data."""
    s = pd.DataFrame()
    s["sex"] = p["sex"]
    s["age"] = age_at_ref(p["dob"])
    s["age_group"] = pd.cut(s["age"], AGE_BINS, right=False, labels=AGE_LABELS).astype(str)
    s["zip"] = p["zip"].astype(str)
    s["education"] = p["education"].map(EDU_GROUP)
    s["citizenship"] = np.where(p["citizenship"] == "Denmark", "Danish", "Other")
    s["marital_status"] = p["marital_status"]
    s["evote"] = p["evote"].astype(int)
    s["party"] = p["party"].replace({"Invalid vote": "Invalid"})
    return s


# ---------------------------------------------------------------- tests

def mc_gof(obs, probs, n_sim=100_000, seed=RNG_SEED):
    """Monte Carlo exact-style chi-square goodness-of-fit (handles small expected counts)."""
    obs = np.asarray(obs, float)
    probs = np.asarray(probs, float) / np.sum(probs)
    n = obs.sum()
    exp = n * probs
    stat = ((obs - exp) ** 2 / exp).sum()
    rng = np.random.default_rng(seed)
    sims = rng.multinomial(int(n), probs, size=n_sim)
    sstat = ((sims - exp) ** 2 / exp).sum(axis=1)
    p = (np.sum(sstat >= stat - 1e-12) + 1) / (n_sim + 1)
    return stat, p


def mc_indep(x, y, n_perm=20_000, seed=RNG_SEED):
    """Chi-square test of independence; Monte Carlo permutation p-value + Cramer's V."""
    m = pd.notna(x) & pd.notna(y)
    x, y = pd.Series(np.asarray(x)[m]), pd.Series(np.asarray(y)[m])
    tab = pd.crosstab(x, y)
    tab = tab.loc[tab.sum(1) > 0, tab.sum(0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return dict(chi2=np.nan, p=np.nan, V=np.nan, n=int(m.sum()))
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    rng = np.random.default_rng(seed)
    xc = pd.Categorical(x).codes
    yc = pd.Categorical(y).codes
    nx, ny = xc.max() + 1, yc.max() + 1
    rs, cs = np.bincount(xc, minlength=nx), np.bincount(yc, minlength=ny)
    exp = np.outer(rs, cs) / len(xc)
    exp_ok = exp > 0
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(yc)
        t = np.bincount(xc * ny + yp, minlength=nx * ny).reshape(nx, ny)
        s = (((t - exp) ** 2)[exp_ok] / exp[exp_ok]).sum()
        cnt += s >= chi2 - 1e-9
    p = (cnt + 1) / (n_perm + 1)
    k = min(tab.shape) - 1
    V = np.sqrt(chi2 / (tab.values.sum() * k))
    return dict(chi2=chi2, p=p, V=V, n=int(tab.values.sum()))


ATTRS = ["sex", "age_group", "zip", "education", "citizenship", "marital_status"]


def analysis(s, results, label=""):
    """Run the analyses for research questions A, B and C. Returns dict of tables."""
    out = {}
    # ---------------- (A) survey vs official results by channel
    rows = []
    channels = {
        "E-vote": (s[s.evote == 1], results.loc["E-votes"]),
        "Polling station": (s[s.evote == 0], results.loc[[i for i in results.index if i.startswith("Polling")]].sum()),
    }
    for ch, (sub, res) in channels.items():
        obs = [(sub.party == k).sum() for k in ["Red", "Green", "Invalid"]]
        pr = [res["Red"], res["Green"], res["Invalid"]]
        chi, p = mc_gof(obs, pr) if obs[2] > 0 else mc_gof(obs[:2], pr[:2])
        n_valid = obs[0] + obs[1]
        bt = stats.binomtest(obs[1], n_valid, res["Green"] / (res["Red"] + res["Green"]))
        rows.append(dict(channel=ch, n=sum(obs), red=obs[0], green=obs[1], invalid=obs[2],
                         survey_green_share=obs[1] / n_valid,
                         official_green_share=res["Green"] / (res["Red"] + res["Green"]),
                         gof_chi2=chi, gof_p=p, binom_p=bt.pvalue))
    out["A"] = pd.DataFrame(rows)
    # per-ZIP polling stations (paper voters only), only if zip present
    if "zip" in s and s["zip"].notna().any():
        zr = []
        for z in ["2100", "2200", "2300", "2400"]:
            sub = s[(s.evote == 0) & (s.zip == z)]
            res = results.loc[f"Polling station: ZIP {z}"]
            obs = [(sub.party == k).sum() for k in ["Red", "Green", "Invalid"]]
            chi, p = mc_gof(obs, [res["Red"], res["Green"], res["Invalid"]])
            zr.append(dict(zip=z, n=sum(obs), survey_green_share=obs[1] / max(1, obs[0] + obs[1]),
                           official_green_share=res["Green"] / (res["Red"] + res["Green"]), gof_p=p))
        out["A_zip"] = pd.DataFrame(zr)
    # channel difference within survey and within official results
    v = s[s.party != "Invalid"]
    out["A_channel_survey"] = mc_indep(v.evote, v.party)
    off = results.loc[["E-votes"]][["Red", "Green"]].copy()
    off.loc["Paper"] = results.loc[[i for i in results.index if i.startswith("Polling")]][["Red", "Green"]].sum()
    c2 = stats.chi2_contingency(off.values, correction=False)
    out["A_channel_official"] = dict(chi2=c2[0], p=c2[1])

    # ---------------- (B) party vs demographics (valid votes)
    rows = []
    attrs = [a for a in ATTRS if a in s.columns]
    for a in attrs + ["evote"]:
        x = v[a].where(v[a] != "Not stated") if a == "education" else v[a]
        r = mc_indep(x, v.party)
        tab = pd.crosstab(x, v.party, normalize="index")
        r.update(attr=a, green_share_by_level="; ".join(f"{i}: {tab.loc[i,'Green']:.0%}" for i in tab.index))
        rows.append(r)
    out["B"] = pd.DataFrame(rows)[["attr", "n", "chi2", "p", "V", "green_share_by_level"]]
    out["B_logit"] = logit(v.assign(y=(v.party == "Green").astype(int)), "y")

    # ---------------- (C) channel vs demographics
    rows = []
    for a in attrs:
        x = s[a].where(s[a] != "Not stated") if a == "education" else s[a]
        r = mc_indep(x, s.evote)
        tab = pd.crosstab(x, s.evote, normalize="index")
        r.update(attr=a, evote_share_by_level="; ".join(f"{i}: {tab.loc[i,1]:.0%}" for i in tab.index))
        rows.append(r)
    out["C"] = pd.DataFrame(rows)[["attr", "n", "chi2", "p", "V", "evote_share_by_level"]]
    out["C_logit"] = logit(s.assign(y=s.evote), "y", exclude_evote=True)
    return out


def logit(d, y, exclude_evote=False):
    d = d.copy()
    d = d[d.education != "Not stated"]
    # suppressed values (NaN) enter the model as their own 'Suppressed' level
    for c in ["sex", "zip", "age_group", "marital_status"]:
        if c in d.columns:
            d[c] = d[c].astype(object).where(d[c].notna(), "Suppressed")
    terms = ["C(sex, Treatment('Female'))", f"C(age_group, Treatment('{sorted(d.age_group.unique())[0]}'))",
             "C(zip, Treatment('2100'))", "C(education, Treatment('Secondary/VET'))"]
    if "marital_status" in d.columns:
        d["marital"] = d["marital_status"].replace({"Widowed": "Divorced/Widowed", "Divorced": "Divorced/Widowed"})
        terms.append("C(marital, Treatment('Married/separated'))")
    if "citizenship" in d.columns:
        terms.append("C(citizenship)")
    if not exclude_evote:
        terms.append("evote")
    m = smf.logit(f"{y} ~ " + " + ".join(terms), d).fit(disp=0)
    tab = pd.DataFrame({"OR": np.exp(m.params), "p": m.pvalues})
    tab = tab.drop(index="Intercept")
    tab.index = [clean_term(i) for i in tab.index]
    return tab


def clean_term(t):
    import re
    t = re.sub(r"C\((\w+)(, Treatment\('[^']*'\))?\)\[T\.([^\]]*)\]", r"\1 = \3", t)
    return t
