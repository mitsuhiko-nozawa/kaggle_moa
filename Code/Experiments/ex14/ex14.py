import warnings, random, os, sys, tqdm, time 
sys.path.append("../")
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

pd.set_option("display.max_columns", 1200)
pd.set_option("display.max_rows", 1200)




# g772, c100, 206クラス、402クラスの分類

train_df = pd.read_csv("../../../Data/Raw/train_features.csv")
test_df = pd.read_csv("../../../Data/Raw/test_features.csv")
pub_test_df = pd.read_csv("../../../Data/Raw/test_features.csv")
drug_df = pd.read_csv("../../../Data/Raw/train_drug.csv")#

y = pd.read_csv("../../../Data/Raw/train_targets_scored.csv")
y_non = pd.read_csv("../../../Data/Raw/train_targets_nonscored.csv")

train_df = train_df.merge(drug_df, on="sig_id")
y_all = pd.concat([y, y_non.drop("sig_id", axis=1)], axis=1)
y = y.merge(drug_df, on='sig_id', how='left') #

TR_SIZE = train_df.shape[0]
TE_SIZE = test_df.shape[0]
train_nonvehicle_index = train_df[train_df["cp_type"] != "ctl_vehicle"].index
test_nonvehicle_index = test_df[test_df["cp_type"] != "ctl_vehicle"].index
# remove cp_type = ctl_vehicle
mask = train_df["cp_type"] != "ctl_vehicle"

#train_df = train_df[mask].drop("cp_type", axis=1).reset_index(drop=True)
#test_df = test_df[test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
#pub_test_df = pub_test_df[pub_test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
y_nonv = y[mask].reset_index(drop=True)#
y_all_nonv = y_all[mask].reset_index(drop=True)

scored = y_nonv.copy()#
y_nonv.drop("drug_id", axis=1, inplace=True)#
y.drop("drug_id", axis=1, inplace=True)#
y = y.drop("sig_id", axis=1).values

#from utils.funcs import run1, run2, run3, metric
from utils.funcs import run1, run2, metric

#==================================== model1 ====================================
weight_path = "weights1"
if weight_path not in os.listdir('.'):
    os.mkdir(weight_path)
param1 = dict(
    BATCH_SIZE=128,
    DEVICE = ('cuda:2' if torch.cuda.is_available() else 'cpu'),
    LEARNING_RATE = 1e-3,
    MAX_LR = 1e-2,
    TR_MAX_LR = 1e-2,
    WEIGHT_DECAY = 1e-5,
    TR_WEIGHT_DECAY = 3e-6,
    DIV_FACTOR = 1e3, 
    TR_DIV_FACTOR = 1e2,
    PCT_START = 0.1,
    EPOCHS = 24,
    EARLY_STOPPING_STEPS = 10,
    K = 7,
    decomp_g = 600,
    decomp_c = 50,
    WEIGHT_DIR = weight_path,
    VT = 0.85,
    kmeans_g = 22,
    kmeans_c = 4,
    kmeans_pca = 5,
    SCALE = "quantile",

)
seeds = [0, 1, 2, 3, 4, 5, 6]
mode = "infer"
train_preds, preds = run1(
    train_df, 
    pub_test_df, 
    test_df, 
    y_nonv, 
    y_all_nonv,
    scored, 
    seeds,
    param1,
    mode,
)
train_preds1 = np.zeros((TR_SIZE,  206))
train_preds1[train_nonvehicle_index] = train_preds
preds1 = np.zeros((TE_SIZE, 206))
preds1[test_nonvehicle_index] = preds

print("cv score : {}".format(metric(y, train_preds1)))

sub_df = pd.read_csv("../../../Data/Raw/sample_submission.csv")
cols = [col for col in sub_df.columns if col != "sig_id"]
sub_df[cols] = preds1
sub_df.to_csv("submission.csv", index=False)


train_sub = pd.read_csv("../../../Data/Raw/train_targets_scored.csv")
cols = [col for col in train_sub.columns if col != "sig_id"]
train_sub[cols] = train_preds1
train_sub.to_csv("train_preds1.csv", index=False)


#==================================== model2 ====================================
weight_path = "weights2"
if weight_path not in os.listdir('.'):
    os.mkdir(weight_path)
param2 = dict(
    BATCH_SIZE=1024,
    V_BATCH_SIZE=32,
    DEVICE = ('cuda:2' if torch.cuda.is_available() else 'cpu'),
    LEARNING_RATE = 1e-3,
    WEIGHT_DECAY = 1e-5,
    EPOCHS = 200,
    EARLY_STOPPING_STEPS = 20,
    K = 5,
    decomp_g = 600,
    decomp_c = 50,
    WEIGHT_DIR = weight_path,
    VT = 0.8,
    SCALE = "quantile",
    tabnet = dict(
        n_d=24, 
        n_a=24, 
        n_steps=1, 
        gamma=1.3, 
        lambda_sparse=0, 
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(
            lr=2e-2
        ), 
        mask_type='entmax', 
        scheduler_fn=torch.optim.lr_scheduler.MultiStepLR,
        scheduler_params=dict(
            milestones=[ 50,100,150], 
            gamma=0.9
        ), 
        seed = 42,
        verbose = 10
    )
)
seeds = [7, 8, 9, 10, 11, 12, 13]
mode = "train"
train_preds, preds = run2(
    train_df, 
    pub_test_df, 
    test_df, 
    y_nonv, 
    y_non,
    y_all_nonv,
    scored, 
    seeds,
    param2,
    mode,
)
train_preds2 = np.zeros((TR_SIZE,  206))
train_preds2[train_nonvehicle_index] = train_preds
preds2 = np.zeros((TE_SIZE, 206))
preds2[test_nonvehicle_index] = preds

print("cv score : {}".format(metric(y, train_preds2)))

train_sub = pd.read_csv("../../../Data/Raw/train_targets_scored.csv")
cols = [col for col in train_sub.columns if col != "sig_id"]
train_sub[cols] = train_preds2
train_sub.to_csv("train_preds2.csv", index=False)

#==================================== model3 ====================================
weight_path = "weights3"
if weight_path not in os.listdir('.'):
    os.mkdir(weight_path)
param3 = dict(
    BATCH_SIZE=124,
    VERBOSE=10,
    DEVICE = ('cuda:2' if torch.cuda.is_available() else 'cpu'),
    LEARNING_RATE = 1e-3,
    WEIGHT_DECAY = 1e-5,
    EPOCHS = 25,
    EARLY_STOPPING_STEPS = 10,
    K = 3,
    decomp_g = 70,
    decomp_c = 10,
    WEIGHT_DIR = weight_path,
    VT = 0.8,
    SEEDS1 = [1, 2, 3, 4, 5, 6, 7],
    SEEDS2 = [8, 9, 10, 11, 12, 13, 14],
    SEEDS3 = [15, 16, 17, 18, 19, 20, 21],
    SEEDS4 = [22, 23, 24, 25, 26, 27, 28],
    SEEDS5 = [29, 30, 31, 32, 33, 34, 35],
    SCALE = "robust"
)
mode = "train"
train_preds, preds = run3(
    train_df, 
    pub_test_df, 
    test_df, 
    y_nonv, 
    y_all_nonv,
    scored, 
    seeds,
    param3,
    mode,
)
train_preds3 = np.zeros((TR_SIZE,  206))
train_preds3[train_nonvehicle_index] = train_preds
preds3 = np.zeros((TE_SIZE, 206))
preds3[test_nonvehicle_index] = preds

print("cv score : {}".format(metric(y, train_preds3)))

train_sub = pd.read_csv("../../../Data/Raw/train_targets_scored.csv")
cols = [col for col in train_sub.columns if col != "sig_id"]
train_sub[cols] = train_preds3
train_sub.to_csv("train_preds3.csv", index=False)

#==================================== submission ====================================
sub_df = pd.read_csv("../../../Data/Raw/sample_submission.csv")
cols = [col for col in sub_df.columns if col != "sig_id"]
sub_df[cols] = (preds1+preds2+preds3)/3
sub_df.to_csv("submission.csv", index=False)




