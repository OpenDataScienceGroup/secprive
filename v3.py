"""v3 anonymisation for group M: combines v1 (ZIP x channel swapping, record-side k, no invalid
category, secret seed) with v2 (k=5, l-diversity checked on released values)."""
import numpy as np
import pandas as pd
from common import *
from anon import recode

REGION = {"2100": "2100-2200", "2200": "2100-2200", "2300": "2300-2400", "2400": "2300-2400"}
COST = {"marital_status": 1.0, "sex": 1.5, "zip": 2.0, "region": 2.0, "age_group": 4.0}


def qi_list(loc, marital):
    return ["sex", "age_group", loc] + (["marital_status"] if marital else [])


def consistency(rel, true, qi):
    """M[i, j]: released record j is consistent with respondent i (same evote; NaN matches anything)."""
    M = true["evote"].values[:, None] == rel["evote"].values[None, :]
    for c in qi:
        t = np.asarray(true[c].values, dtype=object)
        r = np.asarray(rel[c].values, dtype=object)
        M &= pd.isna(r)[None, :] | (t[:, None] == r[None, :])
    return M


def swap_party(party, strata, frac, rng):
    """Invalid votes -> random Red/Green draw from their stratum; then within each stratum a random
    `frac` of records have their party values permuted among themselves."""
    party = np.asarray(party, dtype=object).copy()
    out = party.copy()
    for st in np.unique(strata):
        idx = np.where(strata == st)[0]
        v = party[idx]
        rg = np.array([(v == "Red").sum(), (v == "Green").sum()], float)
        inv = v == "Invalid"
        v[inv] = rng.choice(["Red", "Green"], size=inv.sum(), p=rg / rg.sum())
        party[idx] = v
        sel = idx[rng.random(len(idx)) < frac]
        out[idx] = v
        out[sel] = rng.permutation(party[sel])
    return out


def suppress(rel, true, qi, k, l):
    rel = rel.copy()
    for c in qi:
        rel[c] = rel[c].astype(object)
    col = {c: rel.columns.get_loc(c) for c in qi}
    prio = sorted(qi, key=lambda c: COST[c])
    party = rel["party"].values
    steps = 0
    while True:
        M = consistency(rel, true, qi)
        kt, kr = M.sum(1), M.sum(0)
        div = np.array([len(set(party[m])) for m in M])
        if kt.min() >= k and kr.min() >= k and div.min() >= l:
            return rel, steps
        steps += 1
        if kr.min() < k:  # record-side: blank cheapest remaining attribute of the weakest record
            j = int(np.argmin(kr))
            for c in prio:
                if pd.notna(rel.iat[j, col[c]]):
                    rel.iat[j, col[c]] = np.nan
                    break
            continue
        bad = np.where((kt < k) | (div < l))[0]
        i = bad[np.argmin(kt[bad])]
        cand = (~M[i]) & (rel["evote"].values == true["evote"].values[i])
        if div[i] < l:
            cand &= ~np.isin(party, list(set(party[M[i]])))
        best = None
        for j in np.where(cand)[0]:
            diff = [c for c in qi if pd.notna(rel.iat[j, col[c]]) and rel.iat[j, col[c]] != true.iat[i, true.columns.get_loc(c)]]
            cost = sum(COST[c] for c in diff) + 0.01 * rel.iloc[j][qi].isna().sum()
            if best is None or cost < best[0]:
                best = (cost, j, diff)
        _, j, diff = best
        for c in diff:
            rel.iat[j, col[c]] = np.nan


def build(s, loc="zip", marital=True, k=5, l=2, frac=0.3, seed=None):
    rng = np.random.default_rng(seed)
    a = recode(s)
    if loc == "region":
        a["region"] = a["zip"].map(REGION)
    if not marital:
        a = a.drop(columns="marital_status")
    true = a.copy()
    strata = (a["zip"].astype(str) + "_" + a["evote"].astype(str)).values  # true ZIP x channel
    a["party"] = swap_party(a["party"].values, strata, frac, rng)
    qi = qi_list(loc, marital)
    rel, steps = suppress(a, true, qi, k, l)
    if loc == "region":
        rel = rel.drop(columns="zip")
    cols = ["sex", "age_group", loc, "education"] + (["marital_status"] if marital else []) + ["evote", "party"]
    order = rng.permutation(len(rel))
    return rel[cols], true, order, steps
