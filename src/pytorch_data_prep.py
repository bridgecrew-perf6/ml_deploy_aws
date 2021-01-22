# Data Exploration and Folder Structure creation
# https://github.com/MLWhiz/data_science_blogs/blob/master/compvisblog/Boats_DataExploration.ipynb
###################################################
import torch
import argparse
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None  # default='warn'

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image

from timeit import default_timer as timer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobname', default='', help='job name')
    parser.add_argument('--original_data_dir', default='', help='image foler. Folder contains folders of images with folder name as class name')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='portion of smallest class size to be used as test/val set. Default: 0.2 used for test, 0.2 used for val, 0.6 used for train')
    opt = parser.parse_args()

    #### Initialize variables
    jobname=opt.jobname
    original_data_dir=opt.original_data_dir
    test_ratio=opt.test_ratio

# Data Exploration and Folder Structure creation
# https://github.com/MLWhiz/data_science_blogs/blob/master/compvisblog/Boats_DataExploration.ipynb
###################################################
import torch
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None  # default='warn'

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image

from timeit import default_timer as timer

categories = glob(original_data_dir+"*")

filedf = pd.DataFrame()
for cat in categories:
    files = glob(cat+"/*")
    tempdf = pd.DataFrame({'filepath':files,'category':cat.split("/")[-1]})
    filedf = pd.concat([filedf,tempdf])
    
# print(filedf.category.unique())

gby_cnt = filedf.groupby("category").aggregate('count').rename(columns = {'filepath':'cnt'}).reset_index().sort_values(by='cnt',ascending=False)
# print(gby_cnt)

X_train_val, X_test, _, _ = train_test_split(
        filedf, filedf['category'],stratify=filedf['category'], test_size=test_ratio)

X_train, X_val, _, _ = train_test_split(
        X_train_val, X_train_val['category'], stratify=X_train_val['category'], test_size=0.25)

# print(X_train.head())
X_train['type'] = 'train'
X_val['type'] = 'val'
X_test['type'] = 'test'

fulldf = pd.concat([X_train,X_test,X_val])
# print(fulldf.type.value_counts())
# print(fulldf.head())
fulldf.to_csv(f'src/output/dataPrepFullDf_{jobname}.csv',index=None)   

# new_data_dir = "data/processed" #New data directory. A new folder structure with train, validation and test images will be created.
new_data_dir = "data/processedSmall"
Path(new_data_dir).mkdir(parents=True, exist_ok=True)
new_data_folder = new_data_dir.split('/')[-2]
# os.chdir(f'{new_data_folder}')

# print(fulldf.category.unique())
for cat in fulldf.category.unique():
    Path(f'{new_data_dir}/train/{cat}').mkdir(parents=True, exist_ok=True)
    Path(f'{new_data_dir}/test/{cat}').mkdir(parents=True, exist_ok=True)
    Path(f'{new_data_dir}/val/{cat}').mkdir(parents=True, exist_ok=True)

for i,row in fulldf.iterrows():
    # Boat category
    cat = row['category']
    # section is train,val or test
    section = row['type']
    # input filepath to copy
    ipath = row['filepath']
    # print('ipath',ipath)
    # output filepath to paste
    opath = ipath.replace(original_data_dir,f"{new_data_dir}/{section}/")
    # print('opath',opath)
    # running the cp command
    os.system(f"cp '{ipath}' '{opath}'")

######
# python src/pytorch_data_prep.py  --jobname vl_test1 --original_data_dir 'data/raw/train/train/train' --test_ratio 0.200000
