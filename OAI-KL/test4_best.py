import os, re, argparse
import numpy as np, pandas as pd, cv2, natsort
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import ttach, torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model4 import model_return

ap=argparse.ArgumentParser()
ap.add_argument('-m','--model_type'); ap.add_argument('-i','--image_size',type=int)
a=ap.parse_args()
sz=(a.image_size,a.image_size)
md=f'./models4/{a.model_type}/{sz}'
test=pd.read_csv('./KneeXray4/test/test_correct.csv')
y=test['label'].values; N=len(y)
tr=A.Compose([A.Resize(sz[0],sz[1],interpolation=cv2.INTER_CUBIC,p=1),
    A.Normalize([0.485,.456,.406],[.229,.224,.225]),ToTensorV2()])
loader=DataLoader(ImageDataset(test,transforms=tr),batch_size=8,shuffle=False)
tt=ttach.Compose([ttach.HorizontalFlip()])
# last (lowest val-loss) checkpoint per fold
ckpts={}
for f in range(1,6):
    eps=[int(re.match(r'%dfold_epoch(\d+)\.pt'%f, x).group(1)) for x in os.listdir(md) if re.match(r'%dfold_epoch(\d+)\.pt'%f, x)]
    ckpts[f]=max(eps)
print('model',a.model_type,'chosen epochs',ckpts)
folds_soft=[]
for f,e in ckpts.items():
    net=model_return(a); net.load_state_dict(torch.load(f'{md}/{f}fold_epoch{e}.pt')); net.eval().cuda()
    net=ttach.ClassificationTTAWrapper(net,tt)
    preds=[]; soft=[]
    with torch.no_grad():
        for b in loader:
            out=net(b['image'].cuda())
            preds+=torch.argmax(out,1).tolist()
            soft.append(nn.Softmax(1)(out).cpu().numpy())
    soft=np.concatenate(soft); folds_soft.append(soft)
    print(f'  fold{f} epoch{e} test acc {np.mean(np.array(preds)==y):.4f}')
os.makedirs('./submission4_best',exist_ok=True)
np.save(f'./submission4_best/{a.model_type}_{a.image_size}.npy', np.stack(folds_soft))
print('saved', f'./submission4_best/{a.model_type}_{a.image_size}.npy')
