import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
rng=np.random.default_rng(42)
y=rng.integers(0,2,100)
oof1=np.clip(.2+.6*y+rng.normal(0,.25,100),0,1)
oof2=np.clip(.25+.5*y+rng.normal(0,.2,100),0,1)
print('illustrative OOF AUC',roc_auc_score(y,.5*oof1+.5*oof2))
sample=pd.DataFrame({'id':[9,2,5],'target':0.})
a=pd.DataFrame({'id':[2,5,9],'p1':[.2,.4,.8]})
b=pd.DataFrame({'id':[5,9,2],'p2':[.6,.7,.3]})
assert all(df.id.notna().all() and df.id.is_unique for df in (sample,a,b))
assert set(sample.id)==set(a.id)==set(b.id)
z=sample[['id']].merge(a,on='id',validate='one_to_one').merge(b,on='id',validate='one_to_one')
assert z.id.tolist()==sample.id.tolist()
submission=z.assign(target=.5*z.p1+.5*z.p2)[sample.columns]
assert np.isfinite(submission.target).all() and submission.target.between(0,1).all()
Path('outputs').mkdir(exist_ok=True)
submission.to_csv('outputs/submission.csv',index=False)
check = pd.read_csv('outputs/submission.csv')
assert check.columns.tolist()==sample.columns.tolist()
assert check.id.tolist()==sample.id.tolist()
assert np.isfinite(check.target).all() and check.target.between(0,1).all()
print(submission)
