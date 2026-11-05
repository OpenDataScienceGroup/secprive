import os, json, numpy as np, pandas as pd, common
from common import *; import v3
v3.COST["sex"] = 1.5
s = standardise_raw(load_private()); res = load_results()
seed_file = "/home/claude/proj/SECRET_seed_v3.txt"   # never share
if not os.path.exists(seed_file): open(seed_file, "w").write(str(int.from_bytes(os.urandom(8), "big")))
seed = int(open(seed_file).read())
rel, true, order, steps = v3.build(s, "region", False, k=5, l=2, frac=0.3, seed=seed)
rel.to_csv("v3_aligned_PRIVATE.csv", index=False, na_rep="NA")        # aligned with raw order: internal only
rel.iloc[order].reset_index(drop=True).to_csv("anonymised_dataM_v3.csv", index=False, na_rep="NA")
print("steps", steps, "blanks", rel.isna().sum().to_dict(), "party", rel.party.value_counts().to_dict())
print("party values changed vs truth:", int((rel.party.values != s.party.values).sum()))
