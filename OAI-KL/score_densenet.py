import os, re
import numpy as np, pandas as pd
import cv2, albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch, torch.nn as nn, ttach
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from dataset import ImageDataset
from model import model_return

model_type, image_size = 'densenet_161', 512
md = f'./models/{model_type}/({image_size}, {image_size})'
test_csv = pd.read_csv('./KneeXray/test/test_correct.csv')
test_y = test_csv['label'].values
tr = A.Compose([A.Resize(image_size,image_size,interpolation=cv2.INTER_CUBIC,p=1),
    A.Normalize([0.485,.456,.406],[.229,.224,.225]), ToTensorV2()])
loader = DataLoader(ImageDataset(test_csv, transforms=tr), batch_size=8, shuffle=False)

files = os.listdir(md)
per_fold = {}
for f in files:
    m = re.match(r'([1-5])fold_epoch(\d+)\.pt', f)
    if not m: continue
    fold=int(m.group(1)); ep=int(m.group(2))
    per_fold.setdefault(fold, []).append((ep, f))
# best per fold = highest epoch (= lowest val loss among saves)
best_ckpt={}
for fd,lst in per_fold.items():
    ep,f = max(lst); best_ckpt[fd]=(ep,f)
print('per-fold best checkpoints (highest epoch/lowest val-loss):')

fold_accs=[]; softs=[]
for fd in range(1,6):
    ep,f = best_ckpt[fd]; print('  fold',fd,'epoch',ep)
    net = model_return(type('A',(),{'model_type':model_type,'image_size':image_size})())
    net.load_state_dict(torch.load(f'{md}/{f}'))
    net.eval().cuda()
    net = ttach.ClassificationTTAWrapper(net, ttach.Compose([ttach.HorizontalFlip()]))
    preds=[]; soft=[]
    with torch.no_grad():
        for batch in loader:
            im=batch['image'].cuda(); target=batch['target']
            out=net(im)
            preds += torch.argmax(out,1).tolist()
            soft.append(nn.functional.softmax(out,1).cpu().numpy())
    soft=np.concatenate(soft); 
    a=accuracy_score(test_y,preds); fold_accs.append(a); softs.append(soft)
    print(f'    fold {fd} test acc {a:.4f}  f1macro {f1_score(test_y,preds,average="macro"):.4f}')
    del net; torch.cuda.empty_cache()

ms=np.mean(np.stack(softs),axis=0)
comb=ms.argmax(1)
ens_acc=accuracy_score(test_y,comb); ens_f1=f1_score(test_y,comb,average='macro')
print('\n=== densenet_161 @512 (5-fold best, TTA soft-vote ensemble) ===')
print('mean per-fold acc %.4f | ENSEMBLE acc %.4f f1macro %.4f'%(np.mean(fold_accs), ens_acc, ens_f1))
