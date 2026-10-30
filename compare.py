import pickle, numpy as np, pandas as pd, common, anon, json
from common import *; from anon import *
raw = pickle.load(open("raw_results.pkl","rb")); fin = pickle.load(open("anon_results.pkl","rb"))
res = load_results()
# 1) re-read released CSV and re-run: must equal stored results
c = pd.read_csv("anonymised_dataM.csv", dtype={"zip":"string"})
c["zip"] = c["zip"].astype(object).where(c["zip"].notna(), np.nan)
print(c.party.value_counts(dropna=False).to_dict(), c.isna().sum().to_dict(), c.shape)
re = analysis(c, res)
assert np.allclose(re["B"].p.values, fin["anon"]["B"].p.values) and np.allclose(re["C"].p.values, fin["anon"]["C"].p.values)
print("CSV reproduces analysis")
# 2) comparison table
def row(name, r, a, rs):
    return dict(test=name, raw=r, raw_schema=rs, anon=a)
rows=[]
for i,ch in enumerate(["E-vote","Polling station"]):
    rows.append(row(f"A: survey vs official, {ch} (binomial)", raw["A"].binom_p[i], fin["anon"]["A"].binom_p[i], fin["raw_schema"]["A"].binom_p[i]))
for t in ("B","C"):
    R=raw[t].set_index("attr"); A=fin["anon"][t].set_index("attr"); S=fin["raw_schema"][t].set_index("attr")
    for at in R.index:
        rows.append(row(f"{t}: {at}", R.p[at], A.p.get(at,np.nan), S.p.get(at,np.nan)))
cmp = pd.DataFrame(rows); print(cmp.round(4).to_string())
# 3) stability over 100 independent swap draws (same suppression)
common.mc_indep.__defaults__=(2000, common.RNG_SEED); common.mc_gof.__defaults__=(20000, common.RNG_SEED)
s=standardise_raw(load_private()); rec=recode(s); st=strata_of(rec); sup=local_suppression(rec,3)
keys=["A_E-vote","A_Polling station","B_sex","B_age_group","B_zip","B_education","B_marital_status","B_evote"]
sig={k:[] for k in keys}; accs=[]
for sd in range(100):
    b=sup.copy(); b["party"],pi=pram(b["party"].values,0.7,np.random.default_rng(5000+sd),st)
    o=analysis(b,res)
    sig["A_E-vote"].append(o["A"].binom_p[0]); sig["A_Polling station"].append(o["A"].binom_p[1])
    Bp=o["B"].set_index("attr").p
    for k in keys[2:]: sig[k].append(Bp[k[2:]])
stab={k:dict(share_sig=float((np.array(v)<.05).mean()), median_p=float(np.median(v))) for k,v in sig.items()}
print(json.dumps(stab,indent=0))
pickle.dump(dict(cmp=cmp,stab=stab),open("compare.pkl","wb"))
