import os, re, itertools, json
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

test = pd.read_csv('./KneeXray/test/test_correct.csv')
y = test['label'].values; N = len(y)

MODELS = {'densenet_161':512,'efficientnet_b5':448,'efficientnet_v2_s':456,'regnet_y_8gf':384,
 'resnet_101':384,'resnext_50_32x4d':512,'wide_resnet_50_2':456,'shufflenet_v2_x2_0':512}

def load(p):
    c = pd.read_csv(p); return c['label'].values, c[['prob_0','prob_1','prob_2','prob_3','prob_4']].values

best = {}
for mt, sz in MODELS.items():
    d = f'./submission/{mt}/({sz}, {sz})'; byfold = {f:[] for f in range(1,6)}
    for fn in os.listdir(d):
        m = re.match(r'([1-5])fold_epoch(\d+)_submission\.csv', fn)
        if not m: continue
        lab, pr = load(os.path.join(d, fn))
        byfold[int(m.group(1))].append((accuracy_score(y, lab), int(m.group(2)), pr))
    sel = [max(byfold[f], key=lambda t:(t[0], -t[1])) for f in range(1,6) if byfold[f]]
    best[mt] = np.mean(np.stack([p for _,_,p in sel]), axis=0)   # (N,5)

names = list(best.keys())

def vote(members):
    P = np.stack([best[m] for m in members]); soft = P.mean(0); sp = soft.argmax(1)
    L = np.stack([best[m].argmax(1) for m in members])
    hard=[]; mix=[]
    for i in range(N):
        c=Counter(L[:,i]); mc=max(c.values())
        if list(c.values()).count(mc)>=2:
            hv=min([k for k,v in c.items() if v==mc]); mix.append(int(sp[i]))
        else:
            hv=Counter(L[:,i]).most_common(1)[0][0]; mix.append(hv)
        hard.append(hv)
    hard=np.array(hard); mix=np.array(mix)
    return (accuracy_score(y,sp), accuracy_score(y,hard), accuracy_score(y,mix))

print('=== FULL 8-model ===')
s,h,m=vote(names)
print(f'soft {s:.4f} hard {h:.4f} mix {m:.4f}')

results=[]
for k in range(2,9):
    for combo in itertools.combinations(names,k):
        s,h,m=vote(list(combo)); results.append((m,s,h,k,combo))
results.sort(reverse=True)
print('\n=== TOP 12 subsets by MIX voting ===')
for m,s,h,k,c in results[:12]:
    print(f'mix {m:.4f} | soft {s:.4f} | hard {h:.4f} | k={k} | {[x for x in c]}')

# best per k
print('\n=== BEST per k (by mix) ===')
for k in range(2,9):
    r=[x for x in results if x[3]==k][0]
    print(f'k={k}: mix {r[0]:.4f} | models {list(r[4])}')

best_result=max(results,key=lambda x:x[0])
json.dump({'best_mix_acc':best_result[0],'best_soft_acc':best_result[1],'best_hard_acc':best_result[2],
           'models':list(best_result[4]),'k':best_result[3],
           'all_ranked':[{'mix':float(r[0]),'soft':float(r[1]),'hard':float(r[2]),'k':r[3],'models':list(r[4])} for r in results[:25]]},
          open('/tmp/subset_results.json','w'), indent=1)
print('\nBEST OVERALL:',best_result[0],list(best_result[4]))
