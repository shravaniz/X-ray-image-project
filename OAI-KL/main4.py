# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
from torch import nn, optim
# from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from dataset import ImageDataset
from early_stop4 import EarlyStopping
from model4 import model_return
# from my_custom_loss import my_ce_mse_loss

def train_for_kfold(model, dataloader, criterion, optimizer, scheduler, fold, epoch):
    train_loss = 0.0
    model.train() # Model을 Train Mode로 변환 >> Dropout Layer 같은 경우 Train시 동작 해야 함
    with torch.set_grad_enabled(True): # with문 : 자원의 효율적 사용, 객체의 life cycle을 설계 가능, 항상(True) gradient 연산 기록을 추적
        for batch in tqdm(dataloader, desc=f'Fold {fold} Epoch {epoch} Train', unit='Batch'):
            optimizer.zero_grad() # 반복 시 gradient(기울기)를 0으로 초기화, gradient는 += 되기 때문
            image, labels = batch['image'].cuda(), batch['target'].cuda() # Tensor를 GPU에 할당
            
            # labels = F.one_hot(labels, num_classes=5).float() # nn.MSELoss() 사용 시 필요
            output = model(image) # image(data)를 model에 넣어서 hypothesis(가설) 값을 획득
            
            loss = criterion(output, labels) # Error, Prediction Loss 계산
            train_loss += loss.item() # loss.item()을 통해 Loss의 스칼라 값을 가져온다.

            loss.backward() # Prediction Loss를 Back Propagation으로 계산
            optimizer.step() # optimizer를 이용해 Loss를 효율적으로 최소화 할 수 있게 Parameter 수정
        
        if scheduler is not None:
            scheduler.step()
            
    return train_loss

def val_for_kfold(model, dataloader, criterion, fold, epoch, num_classes=4):
    val_loss = 0.0
    all_preds, all_trues = [], []
    model.eval() # Model을 Eval Mode로 전환 >> Dropout Layer 같은 경우 Eval시 동작 하지 않아야 함
    with torch.no_grad(): # gradient 연산 기록 추적 off
        for batch in tqdm(dataloader, desc=f'Fold {fold} Epoch {epoch} Valid', unit='Batch'):
            image, labels = batch['image'].cuda(), batch['target'].cuda()
            
            output = model(image)
            
            loss = criterion(output, labels)
            val_loss += loss.item()
            
            all_preds.extend(torch.argmax(output, dim=1).cpu().tolist())
            all_trues.extend(labels.cpu().tolist())
             
    acc = accuracy_score(all_trues, all_preds)
    f1m = f1_score(all_trues, all_preds, average='macro', zero_division=0)
    f1w = f1_score(all_trues, all_preds, average='weighted', zero_division=0)
    return val_loss, acc, f1m, f1w, all_preds, all_trues

def train(train_dataset, val_dataset, args, batch_size, epochs, k, splits, labels, foldperf):
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)), labels), start=1):
        # Data Load에 사용되는 index, key의 순서를 지정하는데 사용, Sequential , Random, SubsetRandom, Batch 등 + Sampler
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)        
        # Data Load
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        
        model_ft = model_return(args)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Loss Function
        # criterion = nn.MSELoss()
        # criterion = my_ce_mse_loss
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.01) # Optimizer
        scheduler = None
        
        if torch.cuda.device_count() > 1:
            model_ft = nn.DataParallel(model_ft) # model이 여러 대의 gpu에 할당되도록 병렬 처리
        model_ft.cuda() # Model을 GPU에 할당
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1_macro': [], 'val_f1_weighted': []}
            
        patience = 7
        delta = 0.1
        early_stopping = EarlyStopping(args, patience=patience, verbose=True, delta=delta)
        
        for epoch in range(1, epochs + 1):
            if epoch == 2:
                for param in model_ft.parameters():
                    param.requires_grad=True

                optimizer = optim.Adam(filter(lambda p: p.requires_grad,model_ft.parameters()), weight_decay=0.0001, lr=0.001)
                # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
                scheduler = MultiStepLR(optimizer, milestones=[2], gamma=0.1)

            print(f"Learning Rate : {optimizer.param_groups[0]['lr']}")
                
            train_loss = train_for_kfold(model_ft, train_loader, criterion, optimizer, scheduler, fold, epoch)
            val_loss, val_acc, val_f1m, val_f1w, vpreds, vtrues = val_for_kfold(model_ft, val_loader, criterion, fold, epoch)
            
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            print(f"Epoch: {epoch}/{epochs} \t Avg Train Loss: {train_loss:.3f} \t Avg Valid Loss: {val_loss:.3f} "
                  f"\t Val Acc: {val_acc:.4f} \t Val F1(macro): {val_f1m:.4f} \t Val F1(weighted): {val_f1w:.4f}")
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1_macro'].append(val_f1m)
            history['val_f1_weighted'].append(val_f1w)
            print(classification_report(vtrues, vpreds, digits=4, zero_division=0))
            
            early_stopping(val_loss, model_ft, args, fold, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        foldperf[f"fold{fold}"] = history
            
    tl_f, vall_f = [], []

    for f in range(1, k+1):
        tl_f.append(np.mean(foldperf[f'fold{f}']['train_loss']))
        vall_f.append(np.mean(foldperf[f'fold{f}']['val_loss']))

    acc_f = [max(foldperf[f'fold{f}']['val_acc']) for f in range(1, k+1)]
    f1m_f = [max(foldperf[f'fold{f}']['val_f1_macro']) for f in range(1, k+1)]
    f1w_f = [max(foldperf[f'fold{f}']['val_f1_weighted']) for f in range(1, k+1)]

    print()
    print(f"Performance of {k} Fold Cross Validation")
    print(f"Avg Train Loss: {np.mean(tl_f):.3f} \t Avg Valid Loss: {np.mean(vall_f):.3f}")
    print(f"BEST-per-fold Valid Accuracy : {np.mean(acc_f):.4f}  (folds: {[f'{a:.4f}' for a in acc_f]})")
    print(f"BEST-per-fold Valid F1(macro): {np.mean(f1m_f):.4f}")
    print(f"BEST-per-fold Valid F1(wtd)  : {np.mean(f1w_f):.4f}")

    import json, os
    os.makedirs('./reports4', exist_ok=True)
    out = {'model': args.model_type, 'image_size': args.image_size,
           'best_fold_acc': acc_f, 'mean_best_acc': float(np.mean(acc_f)),
           'mean_best_f1_macro': float(np.mean(f1m_f)), 'mean_best_f1_weighted': float(np.mean(f1w_f)),
           'history': foldperf}
    with open(f'./reports4/{args.model_type}_{args.image_size}.json', 'w') as fp:
        json.dump(out, fp, indent=1, default=float)
    print(f"history saved -> ./reports4/{args.model_type}_{args.image_size}.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', dest='model_type', action='store')
    parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action='store')
    args = parser.parse_args()
    
    image_size_tuple = (args.image_size, args.image_size)
    
    print(f"Model Type : {args.model_type}")
    print(f"Image Size : {image_size_tuple}")
        
    train_csv = pd.read_csv('./KneeXray4/train/train.csv')

    train_transform = A.Compose([
                    A.Resize(args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1),
                    # A.RandomCrop(height=int(384*0.8), width=int(384*0.8), p=1),
                    # A.GridDistortion(p=0.5),
                    # A.ElasticTransform(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=20, p=1),
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # -1 ~ 1의 범위를 가지도록 정규화
                    ToTensorV2() # 0 ~ 1의 범위를 가지도록 정규화
                    ])
    val_transform = A.Compose([
                    A.Resize(args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1),
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # -1 ~ 1의 범위를 가지도록 정규화
                    ToTensorV2() # 0 ~ 1의 범위를 가지도록 정규화
                    ])
    train_dataset = ImageDataset(train_csv, transforms=train_transform)
    val_dataset = ImageDataset(train_csv, transforms=val_transform)
    
    batch_size = 16
    epochs = 30
    k = 5
    torch.manual_seed(42)
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = train_dataset.get_labels()
    foldperf = {}

    train(train_dataset, val_dataset, args, batch_size, epochs, k, splits, labels, foldperf)