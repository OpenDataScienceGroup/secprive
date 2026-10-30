"""Step 2: anonymisation pipeline + disclosure-risk metrics for group M."""
import numpy as np
import pandas as pd
from common import *

QI = ["sex", "age_group", "zip", "evote", "marital_status"]
# suppression priority: first suppress the attributes least useful for the analyses
SUPP_ORDER = ["marital_status", "zip", "sex", "age_group"]
PARTIES = ["Red", "Green", "Invalid"]


def recode(s):
    a = s.copy()
    a["marital_status"] = np.where(a.marital_status == "Married/separated", "Married/separated", "Not married")
    a["age_group"] = pd.cut(a["age"], [0, 40, 60, 200], right=False, labels=["20-39", "40-59", "60+"]).astype(str)
    a = a.drop(columns=["age", "citizenship"])
    return a


def anonymity_sets(released, true):
    """k_i = number of released records consistent with respondent i's TRUE quasi-identifiers
    (a suppressed value, NaN, is consistent with anything)."""
    R = released[QI].astype(object).values
    T = true[QI].astype(object).values
    n = len(R)
    k = np.zeros(n, int)
    for i in range(n):
        ok = np.ones(n, bool)
        for q in range(len(QI)):
            col = R[:, q]
            ok &= pd.isna(col) | (col == T[i, q])
        k[i] = ok.sum()
    return k


def consistency(released, true):
    """M[i, j] = released record j is consistent with respondent i's true QIs."""
    R = released[QI].astype(object).values
    T = true[QI].astype(object).values
    M = np.ones((len(T), len(R)), bool)
    for q in range(len(QI)):
        M &= pd.isna(R[None, :, q]) | (R[None, :, q] == T[:, None, q])
    return M


def local_suppression(a, k_target):
    """Enforce two conditions (adversary knows the respondents and their true QIs):
    target-side : every respondent is consistent with >= k released records
    record-side : every released record is consistent with >= k respondents.
    Suppressed values (NaN) match anything. evote is never suppressed."""
    rel = a.copy()
    true = a.copy()
    for col in SUPP_ORDER:
        rel[col] = rel[col].astype(object)
    rank = {c: r for r, c in enumerate(SUPP_ORDER)}
    while True:
        M = consistency(rel, true)
        kt, kr = M.sum(1), M.sum(0)
        if kt.min() >= k_target and kr.min() >= k_target:
            return rel
        if kr.min() < k_target:
            # record-side: suppress the next attribute of the weakest record
            j = int(np.argmin(kr))
            for col in SUPP_ORDER:
                if pd.notna(rel.iat[j, rel.columns.get_loc(col)]):
                    rel.iat[j, rel.columns.get_loc(col)] = np.nan
                    break
            continue
        # target-side: make the cheapest extra record consistent with respondent i
        i = int(np.argmin(kt))
        best = None
        for j in np.where(~M[i])[0]:
            diff = [c for c in QI if pd.notna(rel.iat[j, rel.columns.get_loc(c)])
                    and rel.iat[j, rel.columns.get_loc(c)] != true.iat[i, true.columns.get_loc(c)]]
            if "evote" in diff:
                continue
            cost = (len(diff), sum(rank[c] for c in diff))
            if best is None or cost < best[0]:
                best = (cost, j, diff)
        _, j, diff = best
        for c in diff:
            rel.iat[j, rel.columns.get_loc(c)] = np.nan


def pram_matrix(pi, theta):
    """Invariant PRAM: keep the value with prob. theta, otherwise redraw from marginal pi."""
    pi = np.asarray(pi, float)
    return theta * np.eye(len(pi)) + (1 - theta) * np.tile(pi, (len(pi), 1))


def pram(values, theta, rng, strata=None):
    """Random swapping of party within strata (ZIP x voting channel).
    In each stratum, a random ~(1-theta) share of records is selected and their party values are
    randomly permuted among themselves. Party counts per stratum are preserved exactly; for an
    individual record the released party equals the true one with prob. ~ theta + (1-theta)*share.
    Invalid votes (2 records) are first replaced by a draw from the stratum's Red/Green split."""
    values = np.asarray(values, dtype=object).copy()
    if strata is None:
        strata = np.zeros(len(values), int)
    strata = np.asarray(strata)
    out = values.copy()
    for st in np.unique(strata):
        idx = np.where(strata == st)[0]
        v = values[idx]
        rg = np.array([(v == "Red").sum(), (v == "Green").sum()], float)
        inv = v == "Invalid"
        v[inv] = rng.choice(["Red", "Green"], size=inv.sum(), p=rg / rg.sum())
        values[idx] = v
        sel = idx[rng.random(len(idx)) > theta]
        out[idx] = v
        out[sel] = rng.permutation(values[sel])
    pi = np.array([(values == c).mean() for c in PARTIES])
    return out, pi


def strata_of(a):
    return (a["zip"].astype(str) + "_" + a["evote"].astype(str)).values


def anonymise(s, k_target=3, theta=0.75, seed=None):
    rng = np.random.default_rng(seed)
    a = recode(s)
    strata = strata_of(a)  # true ZIP x channel, computed before suppression
    a = local_suppression(a, k_target)
    a["party"], pi = pram(a["party"].values, theta, rng, strata)
    # shuffle row order so file order carries no information
    a = a.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(drop=True)
    return a, pi


# ---------------------------------------------------------------- risk

def register_frame():
    r = load_register()
    R = pd.DataFrame({"name": r.name, "sex": r.sex, "age": age_at_ref(r.dob), "zip": r.zip.astype(str),
                      "citizenship": np.where(r.citizenship == "Denmark", "Danish", "Other"),
                      "marital_status": np.where(r.marital_status == "Married/separated", "Married/separated", "Not married"),
                      "evote": r.last_voted})
    R["age_group"] = pd.cut(R.age, [0, 40, 60, 200], right=False, labels=["20-39", "40-59", "60+"]).astype(str)
    return R


def population_risk(rel, R):
    """Scenario 1 (assumed in Phase 1): adversary has register but not the respondent list.
    F_i = number of register persons consistent with released record i (NaN = wildcard);
    individual re-identification risk ~ 1/F_i."""
    Rv = R[R.evote < 2]
    F = np.zeros(len(rel), int)
    Rm = Rv[QI].astype(object).values
    for i, row in enumerate(rel[QI].astype(object).values):
        ok = np.ones(len(Rm), bool)
        for q in range(len(QI)):
            if pd.notna(row[q]):
                ok &= Rm[:, q] == row[q]
        F[i] = ok.sum()
    return F


def attribute_attack(rel, true, theta, pi, prior=None):
    """Scenario 2 (stronger, Phase-2-like): adversary knows who took part and their true QIs.
    For each respondent: candidate records = released records consistent with their QIs.
    Bayesian posterior over party using the published PRAM matrix; guess = argmax."""
    P = pram_matrix(pi, theta)  # P[true, released]
    if prior is None:
        prior = pi
    idx = {c: j for j, c in enumerate(PARTIES)}
    R = rel[QI].astype(object).values
    T = true[QI].astype(object).values
    relp = rel["party"].map(idx).values
    guesses, conf = [], []
    for i in range(len(T)):
        ok = np.ones(len(R), bool)
        for q in range(len(QI)):
            ok &= pd.isna(R[:, q]) | (R[:, q] == T[i, q])
        cand = np.where(ok)[0]
        # target is one of the candidates (uniformly); posterior over target's true party
        post = np.zeros(3)
        for j in cand:
            lik = prior * P[:, relp[j]]
            post += lik / lik.sum()
        post /= len(cand)
        guesses.append(PARTIES[int(np.argmax(post))])
        conf.append(post.max())
    g = np.array(guesses)
    correct = g == true["party"].values
    return dict(acc=correct.mean(), n_correct=int(correct.sum()),
                n_conf90=int((np.array(conf) >= 0.9).sum()),
                n_conf90_correct=int(((np.array(conf) >= 0.9) & correct).sum()),
                baseline=(true["party"] == "Green").mean())


def naive_attack(rel, true):
    """Ignore PRAM: take majority released party among candidates (ties -> Green)."""
    R = rel[QI].astype(object).values
    T = true[QI].astype(object).values
    rp = rel["party"].values
    corr = 0
    certain = 0
    certain_ok = 0
    for i in range(len(T)):
        ok = np.ones(len(R), bool)
        for q in range(len(QI)):
            ok &= pd.isna(R[:, q]) | (R[:, q] == T[i, q])
        vals = pd.Series(rp[ok]).value_counts()
        g = vals.index[0] if vals.iloc[0] > vals.get("Green", 0) else "Green"
        corr += g == true["party"].iloc[i]
        if len(vals) == 1:
            certain += 1
            certain_ok += g == true["party"].iloc[i]
    return dict(acc=corr / len(T), homogeneous=certain, homogeneous_correct=certain_ok)
