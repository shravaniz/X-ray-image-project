# Adapted from the paper's make_csv.py (sunwxxpi/Knee-KL-Grade-Classification).
# The paper's version globbed the author's own ./KneeXray/HH_1/ split; this builds
# the same CSVs that main.py / test_auto.py / test_ensemble.py expect
#   ./KneeXray/train/train.csv          <- train + val (6,604 images, paper's CV pool)
#   ./KneeXray/test/test_correct.csv    <- test (1,656 images, paper's held-out set)
# from the Mendeley OAI folder layout (data-dir/{train,val,test}/0..4/*.png).
import argparse
import os
from pathlib import Path

import pandas as pd

def build_csv(class_dirs, out_path):
    data = []
    label = []
    for class_dir in class_dirs:
        for grade in range(5):
            for image_path in sorted((Path(class_dir) / str(grade)).glob('*')):
                if image_path.suffix.lower() in ('.png', '.jpg'):
                    data.append(str(image_path.resolve()))
                    label.append(grade)

    df = pd.DataFrame({'data': data, 'label': label})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../KneeXrayData/ClsKLData/kneeKL299', help='folder containing train/, val/, test/ class subfolders')
    args = parser.parse_args()

    build_csv([f'{args.data_dir}/train', f'{args.data_dir}/val'], './KneeXray/train/train.csv')
    build_csv([f'{args.data_dir}/test'], './KneeXray/test/test_correct.csv')
