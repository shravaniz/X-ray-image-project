import os, argparse
import pandas as pd, cv2, natsort
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import ttach
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model4 import model_return

parser=argparse.ArgumentParser()
parser.add_argument('-m','--model_type',dest='model_type',action='store')
parser.add_argument('-i','--image_size',type=int,default=224,dest='image_size',action='store')
args=parser.parse_args()
image_size_tuple=(args.image_size,args.image_size)
test_csv=pd.read_csv('./KneeXray4/test/test_correct.csv')
model_path=f'./models4/{args.model_type}/{image_size_tuple}'
submission_path=f'./submission4/{args.model_type}/{image_size_tuple}'
os.makedirs(submission_path, exist_ok=True)

model_list=[f for f in os.listdir(model_path) if f.endswith('.pt')]
model_list=natsort.natsorted(model_list)
existing=[f for f in os.listdir(submission_path) if f.endswith('.csv')]

transform=A.Compose([A.Resize(image_size_tuple[0],image_size_tuple[1],interpolation=cv2.INTER_CUBIC,p=1),
    A.Normalize([0.485,.456,.406],[.229,.224,.225]),ToTensorV2()])
loader=DataLoader(ImageDataset(test_csv,transforms=transform),batch_size=1,shuffle=False)
tt=ttach.Compose([ttach.HorizontalFlip()])

for i in model_list:
    base=os.path.splitext(i)[0]
    if f'{base}_submission.csv' in existing: continue
    net=model_return(args); net.load_state_dict(torch.load(f'{model_path}/{i}')); net.eval().cuda()
    net=ttach.ClassificationTTAWrapper(net,tt)
    preds=[]; p0=[];p1=[];p2=[];p3=[]
    with torch.no_grad():
        for b in loader:
            im=b['image'].cuda(); out=net(im)
            preds.extend(torch.argmax(out,1).tolist())
            s=nn.Softmax(dim=1)(out).cpu().numpy()[0]
            p0.append(s[0]);p1.append(s[1]);p2.append(s[2]);p3.append(s[3])
    sub=pd.DataFrame({'data':[x.split('/')[-1] for x in test_csv['data']],'label':preds,
        'prob_0':p0,'prob_1':p1,'prob_2':p2,'prob_3':p3})
    sub.to_csv(f'{submission_path}/{base}_submission.csv',index=False)
    print('saved',base)
