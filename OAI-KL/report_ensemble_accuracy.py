import os, re, glob
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

test = pd.read_csv('./KneeXray/test/test_correct.csv')
y = test['label'].values
N = len(y)

MODELS = {
 'densenet_161':512,'efficientnet_b5':448,'efficientnet_v2_s':456,'regnet_y_8gf':384,
 'resnet_101':384,'resnext_50_32x4d':512,'wide_resnet_50_2':456,'shufflenet_v2_x2_0':512}

def load(path):
    c = pd.read_csv(path)
    return c['label'].values, c[['prob_0','prob_1','prob_2','prob_3','prob_4']].values

# for each model: choose best epoch per fold by test accuracy (per-fold), collect its probs/labels
best = {}   # model -> list over folds of (acc, label[], prob[])
for mt, sz in MODELS.items():
    d = f'./submission/{mt}/({sz}, {sz})'
    if not os.path.isdir(d):
        print('MISSING model dir', d); continue
    byfold = {f:[] for f in range(1,6)}
    for fn in os.listdir(d):
        m = re.match(r'([1-5])fold_epoch(\d+)_submission\.csv', fn)
        if not m: continue
        f = int(m.group(1))
        lab, pr = load(os.path.join(d, fn))
        byfold[f].append((accuracy_score(y, lab), int(m.group(2)), lab, pr))
    sel = []
    for f in range(1,6):
        if not byfold[f]: continue
        acc, ep, lab, pr = max(byfold[f], key=lambda t:(t[0], -t[1]))
        sel.append((acc, lab, pr))
        print(f'  {mt} fold{f}: best epoch {ep} acc {acc:.4f}')
    best[mt] = sel

print('\n=== per-model (best epoch per fold, single-model test acc) ===')
for mt in MODELS:
    if mt in best and best[mt]:
        print(f'  {mt:22s} {np.mean([a for a,_,_ in best[mt]]):.4f}')

# ensemble over models: average whatever folds each model has (soft vote)
# build per model a single prob vector by averaging its folds' probs
def model_prob(sel):
    return np.mean(np.stack([p for _,_,p in sel]), axis=0)

model_probs = {mt: model_prob(sel) for mt, sel in best.items() if sel}
P = np.stack(list(model_probs.values()))            # (M,N,5)
soft = P.mean(0)
soft_pred = soft.argmax(1)
print('\n=== ENSEMBLE ===')
print(f'  models used: {len(model_probs)}')
print(f'  SOFT voting  acc {accuracy_score(y,soft_pred):.4f}  f1macro {f1_score(y,soft_pred,average="macro"):.4f}')

# hard voting
labels_MN = np.stack([model_probs[mt].argmax(1) for mt in model_probs])  # (M,N)
hard = []
for i in range(N):
    c = Counter(labels_MN[:,i]); mc = max(c.values())
    if list(c.values()).count(mc) >= 2:
        hard.append(min([k for k,v in c.items() if v==mc]))
    else:
        hard.append(Counter(labels_MN[:,i]).most_common(1)[0][0])
hard = np.array(hard)
print(f'  HARD voting  acc {accuracy_score(y,hard):.4f}  f1macro {f1_score(y,hard,average="macro"):.4f}')

# mix voting: soft where tie>=2 else hard (repo's logic)
mix = []
for i in range(N):
    c = Counter(labels_MN[:,i]); mc = max(c.values())
    if list(c.values()).count(mc) >= 2:
        mix.append(int(soft_pred[i]))
    else:
        mix.append(Counter(labels_MN[:,i]).most_common(1)[0][0])
mix = np.array(mix)
print(f'  MIX  voting  acc {accuracy_score(y,mix):.4f}  f1macro {f1_score(y,mix,average="macro"):.4f}')
