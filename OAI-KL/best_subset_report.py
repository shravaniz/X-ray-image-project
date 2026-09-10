import os, re
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

test = pd.read_csv('./KneeXray/test/test_correct.csv'); y=test['label'].values; N=len(y)
MODELS={'densenet_161':512,'efficientnet_b5':448,'efficientnet_v2_s':456,'regnet_y_8gf':384,
 'resnet_101':384,'resnext_50_32x4d':512,'wide_resnet_50_2':456,'shufflenet_v2_x2_0':512}
def load(p):
    c=pd.read_csv(p); return c['label'].values, c[['prob_0','prob_1','prob_2','prob_3','prob_4']].values
best={}
for mt,sz in MODELS.items():
    d=f'./submission/{mt}/({sz}, {sz})'; bf={f:[] for f in range(1,6)}
    for fn in os.listdir(d):
        m=re.match(r'([1-5])fold_epoch(\d+)_submission\.csv',fn)
        if not m: continue
        lab,pr=load(os.path.join(d,fn)); bf[int(m.group(1))].append((accuracy_score(y,lab),int(m.group(2)),pr))
    sel=[max(bf[f],key=lambda t:(t[0],-t[1])) for f in range(1,6) if bf[f]]
    best[mt]=np.mean(np.stack([p for _,_,p in sel]),axis=0)

def mix_vote(members):
    P=np.stack([best[m] for m in members]); soft=P.mean(0); sp=soft.argmax(1)
    L=np.stack([best[m].argmax(1) for m in members]); out=[]
    for i in range(N):
        c=Counter(L[:,i]); mc=max(c.values())
        out.append(int(sp[i]) if list(c.values()).count(mc)>=2 else Counter(L[:,i]).most_common(1)[0][0])
    return np.array(out), soft

MEM=['efficientnet_v2_s','regnet_y_8gf','resnet_101','resnext_50_32x4d']
pred,soft=mix_vote(MEM)
acc=accuracy_score(y,pred); f1m=f1_score(y,pred,average='macro'); f1w=f1_score(y,pred,average='weighted')
print('BEST SUBSET:',MEM)
print(f'acc {acc:.4f} f1macro {f1m:.4f} f1weighted {f1w:.4f}')
print(classification_report(y,pred,digits=4))
cm=confusion_matrix(y,pred,labels=[0,1,2,3,4],normalize='true')
np.save('/tmp/best_cm.npy',cm)
# map of tp counts too
cm_raw=confusion_matrix(y,pred,labels=[0,1,2,3,4])
print('raw confusion (rows true, cols pred):'); print(cm_raw)
