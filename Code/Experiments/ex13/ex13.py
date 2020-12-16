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

train_df = train_df.merge(drug_df, on="sig_id")

y = pd.read_csv("../../../Data/Raw/train_targets_scored.csv")
y_non = pd.read_csv("../../../Data/Raw/train_targets_nonscored.csv")
y_all = pd.concat([y, y_non.drop("sig_id", axis=1)], axis=1)
y = y.merge(drug_df, on='sig_id', how='left') #

TR_SIZE = train_df.shape[0]
TE_SIZE = test_df.shape[0]
train_nonvehicle_index = train_df[train_df["cp_type"] != "ctl_vehicle"].index
test_nonvehicle_index = test_df[test_df["cp_type"] != "ctl_vehicle"].index
# remove cp_type = ctl_vehicle
mask = train_df["cp_type"] != "ctl_vehicle"

train_df = train_df[mask].drop("cp_type", axis=1).reset_index(drop=True)
test_df = test_df[test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
pub_test_df = pub_test_df[pub_test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
y_nonv = y[mask].reset_index(drop=True)#

scored = y_nonv.copy()#
y_nonv.drop("drug_id", axis=1, inplace=True)#
y.drop("drug_id", axis=1, inplace=True)#


from utils.funcs import run, metric

weight_path = "dnn_weights"
if weight_path not in os.listdir('.'):
    os.mkdir(weight_path)


param = dict(
    BATCH_SIZE=128,
    DEVICE = ('cuda:2' if torch.cuda.is_available() else 'cpu'),
    LEARNING_RATE = 1e-3,
    WEIGHT_DECAY = 1e-5,
    EPOCHS = 25,
    EARLY_STOPPING_STEPS = 10,
    K = 3,
    decomp_g = 80,
    decomp_c = 10,
    WEIGHT_DIR = weight_path,
    VT = 0.8,
)

seeds = [0]
SCALE = "quantile"
mode = "train"

train_preds, preds = run(
    train_df, 
    pub_test_df, 
    test_df, 
    y_nonv, 
    scored, 
    seeds,
    SCALE,
    param,
    mode,
)

train_preds2 = np.zeros((TR_SIZE,  206))
train_preds2[train_nonvehicle_index] = train_preds

preds2 = np.zeros((TE_SIZE, 206))
preds2[test_nonvehicle_index] = preds

y = y.drop("sig_id", axis=1).values
print("cv score : {}".format(metric(y, train_preds2)))


sub_df = pd.read_csv("../../../Data/Raw/sample_submission.csv")
cols = [col for col in sub_df.columns if col != "sig_id"]
sub_df[cols] = preds2
sub_df.to_csv("submission.csv", index=False)


train_sub = pd.read_csv("../../../Data/Raw/train_targets_scored.csv")
cols = [col for col in train_sub.columns if col != "sig_id"]
train_sub[cols] = train_preds2
train_sub.to_csv("train_preds.csv", index=False)

