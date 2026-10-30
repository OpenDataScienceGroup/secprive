import pandas as pd, pickle
from common import *
from scipy import stats
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
p = load_private(); res = load_results(); s = standardise_raw(p)
out = analysis(s, res)
# extra raw-only: age as continuous
v = s[s.party != "Invalid"]
out["B_age_mw"] = stats.mannwhitneyu(v[v.party=="Green"].age, v[v.party=="Red"].age)
out["C_age_mw"] = stats.mannwhitneyu(s[s.evote==1].age, s[s.evote==0].age)
out["ages"] = dict(green=v[v.party=="Green"].age.median(), red=v[v.party=="Red"].age.median(),
                   e=s[s.evote==1].age.median(), paper=s[s.evote==0].age.median())
pickle.dump(out, open("raw_results.pkl","wb"))
for k, t in out.items(): print("==", k); print(t if not isinstance(t, pd.DataFrame) else t.round(4).to_string())
print(s.age_group.value_counts(), s.education.value_counts())
