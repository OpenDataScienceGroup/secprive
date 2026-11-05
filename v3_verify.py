"""Independent check of the shuffled released file only (does not use v3.py internals except data loading)."""
import numpy as np, pandas as pd, common
from common import *
from scipy import stats
import statsmodels.formula.api as smf
p = load_private(); s = standardise_raw(p); res = load_results()
REG = {2100:"2100-2200",2200:"2100-2200",2300:"2300-2400",2400:"2300-2400"}
age = age_at_ref(p.dob)
T = pd.DataFrame({"sex":p.sex.values, "age_group":pd.cut(age,[0,40,60,200],right=False,labels=["20-39","40-59","60+"]).astype(str).values,
                  "region":p.zip.map(REG).values, "evote":p.evote.values, "party":p.party.values})
rel = pd.read_csv("anonymised_dataM_v3.csv")
QI = ["sex","age_group","region"]
def cons(rel, true):
    M = true.evote.values[:,None]==rel.evote.values[None,:]
    for c in QI:
        r=np.asarray(rel[c].values,dtype=object); t=np.asarray(true[c].values,dtype=object)
        M &= pd.isna(r)[None,:] | (t[:,None]==r[None,:])
    return M
M = cons(rel, T); rp = rel.party.values
out = {}
out["rows"]=len(rel); out["invalid_rows"]=int((rel.party=="Invalid vote").sum()+(rel.party=="Invalid").sum())
out["k_person_min"]=int(M.sum(1).min()); out["k_record_min"]=int(M.sum(0).min()); out["k_person_median"]=float(np.median(M.sum(1)))
out["persons_one_party"]=int(sum(len(set(rp[m]))<2 for m in M))
out["persons_>=80%_one_party"]=int(sum(pd.Series(rp[m]).value_counts(normalize=True).max()>=.8 for m in M))
def eliminate(M):
    M=M.copy(); fixed={}; ch=True
    while ch:
        ch=False
        for j in range(M.shape[1]):
            c=np.where(M[:,j])[0]
            if len(c)==1 and c[0] not in fixed: fixed[c[0]]=j; M[c[0],:]=False; M[:,j]=False; M[c[0],j]=True; ch=True
        for i in range(M.shape[0]):
            r=np.where(M[i])[0]
            if len(r)==1 and i not in fixed: fixed[i]=r[0]; M[:,r[0]]=False; M[i,r[0]]=True; ch=True
    return fixed
out["pinned_by_elimination"]=len(eliminate(M))
out["attacker_majority_acc"]=float(np.mean([pd.Series(rp[M[i]]).value_counts().idxmax()==T.party[i] for i in range(200)]))
out["baseline_always_green"]=float((T.party=="Green").mean())
# outsider: register linkage
R = load_register(); R=R[R.last_voted<2]; ra=age_at_ref(R.dob)
RQ = pd.DataFrame({"sex":R.sex.values,"age_group":pd.cut(ra,[0,40,60,200],right=False,labels=["20-39","40-59","60+"]).astype(str).values,
                   "region":R.zip.map(REG).values,"evote":R.last_voted.values})
F = cons(rel, RQ).sum(0); out["F_min"]=int(F.min()); out["F_median"]=float(np.median(F)); out["sum_1_over_F"]=round(float((1/F).sum()),2)
for k,v in out.items(): print(f"{k:28s} {v}")

# ---------------- utility: raw (same coding) vs v3
raw = T.assign(education=s.education.values, party=s.party.values)
def tests(d):
    r={}
    for ch,sub,off in [("E-vote",d[d.evote==1],res.loc["E-votes"]),("Paper",d[d.evote==0],res.loc[[i for i in res.index if i.startswith("Polling")]].sum())]:
        g=(sub.party=="Green").sum(); n=g+(sub.party=="Red").sum()
        r[f"A {ch}"]=stats.binomtest(g,n,off.Green/(off.Green+off.Red)).pvalue
    v=d[d.party.isin(["Red","Green"])]
    for a in ["sex","age_group","region","education","evote"]:
        x=v[a].where(v[a]!="Not stated") if a=="education" else v[a]
        r[f"B {a}"]=mc_indep(x,v.party)["p"]
    for a in ["sex","age_group","region","education"]:
        x=d[a].where(d[a]!="Not stated") if a=="education" else d[a]
        r[f"C {a}"]=mc_indep(x,d.evote)["p"]
    return r
def region_A(d):
    rows=[]
    for rg,zs in [("2100-2200",[2100,2200]),("2300-2400",[2300,2400])]:
        off=res.loc[[f"Polling station: ZIP {z}" for z in zs]].sum(); sub=d[(d.evote==0)&(d.region==rg)]
        g=(sub.party=="Green").sum(); n=g+(sub.party=="Red").sum()
        rows.append(stats.binomtest(g,n,off.Green/(off.Green+off.Red)).pvalue)
    return rows
def logit(d, y):
    d=d[d.education!="Not stated"].copy()
    for c in ["sex","region"]: d[c]=d[c].astype(object).where(d[c].notna(),"Suppressed")
    if y=="green": d=d[d.party.isin(["Red","Green"])]; d["green"]=(d.party=="Green").astype(int); f="green ~ C(sex,Treatment('Female')) + C(age_group) + C(region) + C(education,Treatment('Secondary/VET')) + evote"
    else: f="evote ~ C(sex,Treatment('Female')) + C(age_group) + C(region) + C(education,Treatment('Secondary/VET'))"
    m=smf.logit(f,d).fit(disp=0); t=pd.DataFrame({"OR":np.exp(m.params),"p":m.pvalues}).drop("Intercept")
    t.index=[common.clean_term(i).replace("C(sex,Treatment('Female'))","sex").replace("C(education,Treatment('Secondary/VET'))","education") for i in t.index]
    return t
a,b=tests(raw),tests(rel)
cmp=pd.DataFrame({"raw":a,"v3":b}); cmp["same conclusion"]=(cmp.raw<.05)==(cmp.v3<.05)
print(cmp.round(4).to_string())
print("A per region, raw:",np.round(region_A(raw),3),"v3:",np.round(region_A(rel),3))
for y in ("green","evote"):
    L=pd.concat([logit(raw,y),logit(rel,y)],axis=1,keys=["raw","v3"]); print(f"logit {y}\n",L.round(3).to_string())
# stability over 30 other swap draws (same pipeline, different random seeds)
import v3; v3.COST["sex"]=1.5; common.mc_indep.__defaults__=(2000, common.RNG_SEED)
agree=[]; accs=[]
for sd in range(30):
    r2,true2,_,_=v3.build(s,"region",False,seed=900+sd)
    t2=tests(r2.assign(**{})); agree.append({k:(t2[k]<.05)==(a[k]<.05) for k in a})
print("share of 30 draws reaching the raw-data conclusion:\n", pd.DataFrame(agree).mean().round(2).to_string())
