import os, json, pickle
import pandas as pd, numpy as np
from common import *; from anon import *
THETA = 0.7
p = load_private(); res = load_results(); s = standardise_raw(p); s0 = recode(s); R = register_frame()
seed_file = "/home/claude/proj/SECRET_seed.txt"   # never published
if not os.path.exists(seed_file): open(seed_file, "w").write(str(int.from_bytes(os.urandom(8), "big")))
seed = int(open(seed_file).read())
a, pi = anonymise(s, 3, THETA, seed)
out_csv = a[["sex", "age_group", "zip", "education", "marital_status", "evote", "party"]].copy()
out_csv.to_csv("anonymised_dataM.csv", index=False, na_rep="NA")
anon_res = analysis(a, res)
# raw data in released schema (recoding only, no suppression/PRAM) -> separates recoding loss from perturbation loss
raw_schema = recode(s); raw_schema_res = analysis(raw_schema, res)

# ---- risk
M = consistency(a, s0)
risk = {}
risk["k_target_min"] = int(M.sum(1).min()); risk["k_record_min"] = int(M.sum(0).min())
risk["k_target_dist"] = pd.Series(M.sum(1)).describe().to_dict()
risk["suppressed"] = {c: int(a[c].isna().sum()) for c in QI}
risk["records_touched"] = int(a[QI].isna().any(axis=1).sum())
risk["changed_party"] = None  # computed below on aligned data
F = population_risk(a, R)
risk["F_min"] = int(F.min()); risk["F_lt5"] = int((F < 5).sum()); risk["F_lt10"] = int((F < 10).sum())
risk["sum_1_over_F"] = float((1 / F).sum()); risk["max_1_over_F"] = float((1 / F).max()); risk["F_median"] = float(np.median(F))
# expected re-identifications if adversary does not know membership: each register person matched w.p. 1/F
# compare: raw data with names removed only
raw_noname = s0.copy(); raw_noname["dob"] = p.dob
Rr = load_register(); Rr = Rr[Rr.last_voted < 2]
key = pd.DataFrame({"sex": Rr.sex, "dob": Rr.dob, "zip": Rr.zip.astype(str), "evote": Rr.last_voted,
                    "marital_status": np.where(Rr.marital_status == "Married/separated", "Married/separated", "Not married")})
g = key.groupby(["sex", "dob", "zip", "evote"]).size()
Fraw = raw_noname.set_index(["sex", "dob", "zip", "evote"]).index.map(g)
risk["raw_noname_F1"] = int((np.array(Fraw) == 1).sum()); risk["raw_noname_sum_1_over_F"] = float((1 / np.array(Fraw)).sum())
# recoded but without suppression/PRAM
rec = recode(s); Fr = population_risk(rec.assign(**{c: rec[c].astype(object) for c in QI}), R)
Mr = consistency(rec, s0)
risk["recoded_only"] = dict(F_lt5=int((Fr < 5).sum()), sum_1_over_F=float((1 / Fr).sum()), k1=int((Mr.sum(1) == 1).sum()), k_lt3=int((Mr.sum(1) < 3).sum()))
# attribute disclosure with leaked respondent list
risk["attack_final"] = attribute_attack(a, s0, THETA, pi)
risk["attack_naive_final"] = naive_attack(a, s0)
no_pram = local_suppression(recode(s), 3)
risk["attack_no_pram"] = naive_attack(no_pram, s0)
risk["attack_recoded_only"] = naive_attack(rec, s0)
risk["pi"] = pi.tolist(); risk["theta"] = THETA
P = pram_matrix(pi, THETA); risk["P"] = P.tolist()
post_G = P[1,1]*pi[1]/(P[:,1]@pi); post_R = P[0,0]*pi[0]/(P[:,0]@pi)
risk["posterior_G"] = float(post_G); risk["posterior_R"] = float(post_R)
risk["eps_G"] = float(np.log(P[1,1]/P[0,1])); risk["eps_R"] = float(np.log(P[0,0]/P[1,0]))
# party changed count: need aligned run -> rerun pipeline without shuffle
rng = np.random.default_rng(seed); _ = local_suppression  # same seed sequence as anonymise
st = strata_of(recode(s)); sup = local_suppression(recode(s), 3); newp, _ = pram(sup["party"].values, THETA, rng, st)
risk["changed_party"] = int((newp != s.party.values).sum())
# robustness: repeat PRAM 200 times, attacker accuracy distribution + utility stability
accs, sig = [], []
for sd in range(200):
    rng2 = np.random.default_rng(10_000 + sd)
    b = no_pram.copy(); b["party"], pi2 = pram(b["party"].values, THETA, rng2, strata_of(recode(s)))
    accs.append(attribute_attack(b, s0, THETA, pi2)["acc"])
risk["attack_acc_mc"] = dict(mean=float(np.mean(accs)), p5=float(np.percentile(accs, 5)), p95=float(np.percentile(accs, 95)))
pickle.dump(dict(anon=anon_res, raw_schema=raw_schema_res, risk=risk, F=F, kt=M.sum(1)), open("anon_results.pkl", "wb"))
print(json.dumps(risk, indent=1, default=float))
for k, t in anon_res.items(): print("==", k); print(t if not isinstance(t, pd.DataFrame) else t.round(4).to_string())
