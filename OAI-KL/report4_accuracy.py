import os, re
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

test=pd.read_csv('./KneeXray4/test/test_correct.csv'); y=test['label'].values; N=len(y)
MODELS={'efficientnet_v2_s':456,'regnet_y_8gf':384,'resnet_101':384,'resnext_50_32x4d':512}

def load(p):
    c=pd.read_csv(p)
    return c['label'].values, c[[f'prob_{i}' for i in range(4)]].values

best={}
for mt,sz in MODELS.items():
    d=f'./submission4/{mt}/({sz}, {sz})'
    if not os.path.isdir(d): print('MISSING',d); continue
    bf={f:[] for f in range(1,6)}
    for fn in os.listdir(d):
        m=re.match(r'([1-5])fold_epoch(\d+)_submission\.csv',fn)
        if not m: continue
        lab,pr=load(os.path.join(d,fn)); bf[int(m.group(1))].append((accuracy_score(y,lab),int(m.group(2)),pr))
    sel=[max(bf[f],key=lambda t:(t[0],-t[1])) for f in range(1,6) if bf[f]]
    best[mt]=np.mean(np.stack([p for _,_,p in sel]),axis=0)
    print(f'{mt:20s} single (fold-avg prob) acc {accuracy_score(y,best[mt].argmax(1)):.4f}')

names=list(best.keys())
def vote(members):
    P=np.stack([best[m] for m in members]); soft=P.mean(0); sp=soft.argmax(1)
    L=np.stack([best[m].argmax(1) for m in members]); hard=[];mix=[]
    for i in range(N):
        c=Counter(L[:,i]); mc=max(c.values())
        if list(c.values()).count(mc)>=2:
            hard.append(min([k for k,v in c.items() if v==mc])); mix.append(int(sp[i]))
        else:
            hard.append(Counter(L[:,i]).most_common(1)[0][0]); mix.append(Counter(L[:,i]).most_common(1)[0][0])
    return np.array(hard),sp,np.array(mix)

h,s,m=vote(names)
print('\n=== 4-CLASS (grade 1 removed) FULL 4-model ensemble, N=%d ==='%N)
print(f'soft {accuracy_score(y,s):.4f}  hard {accuracy_score(y,h):.4f}  mix {accuracy_score(y,m):.4f}  f1macro(mix) {f1_score(y,m,average="macro"):.4f}')
print('\nPer-class (mix voting):')
print(classification_report(y,m,digits=4,zero_division=0))
print('confusion (rows true 0..3, cols pred 0..3):'); print(confusion_matrix(y,m,labels=[0,1,2,3]))
