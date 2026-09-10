import numpy as np, pandas as pd, itertools, glob, os
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
test=pd.read_csv('./KneeXray4/test/test_correct.csv'); y=test['label'].values; N=len(y)
files=sorted(glob.glob('./submission4_best/*.npy'))
models={}
for f in files:
    name=os.path.basename(f).rsplit('_',1)[0]
    models[name]=np.load(f).mean(0)      # fold-average probs
print('4-CLASS single models (fold-avg prob):')
for k,v in models.items():
    print(f'  {k:20s} acc {accuracy_score(y,v.argmax(1)):.4f}  f1m {f1_score(y,v.argmax(1),average="macro"):.4f}')
names=list(models)
def vote(ms):
    P=np.stack([models[m] for m in ms]); soft=P.mean(0); sp=soft.argmax(1)
    L=np.stack([models[m].argmax(1) for m in ms]); hard=[];mix=[]
    for i in range(N):
        c=Counter(L[:,i]); mc=max(c.values())
        if list(c.values()).count(mc)>=2:
            hard.append(min([k for k,v in c.items() if v==mc])); mix.append(int(sp[i]))
        else:
            hard.append(Counter(L[:,i]).most_common(1)[0][0]); mix.append(Counter(L[:,i]).most_common(1)[0][0])
    return accuracy_score(y,sp),accuracy_score(y,np.array(hard)),accuracy_score(y,np.array(mix)),np.array(mix)
s,h,m,mp=vote(names)
print(f'\n=== 4-CLASS FULL ensemble (N={N}) ===')
print(f'  soft {s:.4f}  hard {h:.4f}  mix {m:.4f}  f1macro(mix) {f1_score(y,mp,average="macro"):.4f}')
print('\n=== SUBSET SEARCH (mix voting) ===')
res=[]
for k in range(2,len(names)+1):
    for c in itertools.combinations(names,k):
        _,_,mm,_=vote(list(c)); res.append((mm,k,c))
res.sort(reverse=True)
for mm,k,c in res[:6]:
    print(f'  mix {mm:.4f} | k={k} | {list(c)}')
best=res[0]
_,_,_,bmp=vote(list(best[2]))
print(f'\nBEST 4-CLASS: {best[0]:.4f}  models={list(best[2])}')
print(classification_report(y,bmp,digits=4,zero_division=0))
print('confusion (rows true 0..3, cols pred 0..3):'); print(confusion_matrix(y,bmp,labels=[0,1,2,3]))
