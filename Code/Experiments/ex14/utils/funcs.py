import warnings, random, os, sys, tqdm, time
sys.path.append("../")
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
from .torch_utils import run_model1#, run_model2, run_model3
from .prepro import prepro1#, prepro2, prepro3


def metric(y_true, y_pred):
    res = []
    for i in range(0, y_true.shape[1]):
        y = y_true[:,i]
        pred = y_pred[:,i]
        res.append(log_loss(y, pred))
    return np.mean(res)

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
seed_everything(42)
        
    
def make_folds(targets, scored, seed, K):
    # LOCATE DRUGS
    vc = scored["drug_id"].value_counts()
    vc1 = vc.loc[vc<=18].index.sort_values()
    vc2 = vc.loc[vc>18].index.sort_values()

    # STRATIFY DRUGS 18X OR LESS
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    tmp = scored.groupby('drug_id')[targets].mean().loc[vc1]
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.index[idxV].values} # drug id がどのフォールドに属すか格納
        dct1.update(dd)

    # STRATIFY DRUGS MORE THAN 18X
    skf = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    tmp = scored.loc[scored["drug_id"].isin(vc2)].reset_index(drop=True)
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp["sig_id"][idxV].values}
        dct2.update(dd)

    # ASSIGN K
    scored['fold'] = scored.drug_id.map(dct1)
    scored.loc[scored["fold"].isna(),'fold'] = scored.loc[scored["fold"].isna(),'sig_id'].map(dct2)
    scored["fold"] = scored["fold"].astype('int8')
    return scored


def train_by_fold1(X, y_nonv, y_all_nonv, pub_test_X, test_X, seed, param, scored, fold, mode):
    train_index = scored[scored["fold"] != fold].index.to_list()
    valid_index = scored[scored["fold"] == fold].index.to_list()

    train_X = X.iloc[train_index]
    train_y = y_nonv[train_index]
    train_y_all = y_all_nonv[train_index]
    valid_X = X.iloc[valid_index]
    valid_y = y_nonv[valid_index]
    valid_y_all = y_all_nonv[valid_index]

    # prepare data for training
    train_X = train_X.values
    valid_X = valid_X.values
    test_X = test_X.values
    print(train_X.shape)

    val_preds, test_preds = run_model1(train_X, train_y, train_y_all, valid_X, valid_y, valid_y_all, test_X, param, seed, fold, mode)
    return valid_index, val_preds, test_preds


def train_by_seed1(X, y_nonv, y_all_nonv, pub_test_X, test_X, scored, seed, param, mode):
    seed_everything(seed)
    K = param["K"]
    train_preds = np.zeros((X.shape[0], y_nonv.shape[1]))
    preds_seed = np.zeros((test_X.shape[0], y_nonv.shape[1]))
    
    targets = [col for col in y_nonv.columns if col != "sig_id" and col != "drug_id"]
    y_nonv = y_nonv.values
    scored = make_folds(targets, scored, seed, K)

    for fold in range(K):
        val_index, val_preds, preds = train_by_fold1(X, y_nonv, y_all_nonv, pub_test_X, test_X, seed, param, scored, fold, mode)
        train_preds[val_index] = val_preds
        preds_seed += preds
    preds_seed /= K
    print("cv score : {}".format(metric(y_nonv, train_preds)))
    return train_preds, preds_seed


def run1(train_df, pub_test_df, test_df, y_nonv, y_all_nonv, scored, seeds, param, mode):
    train_df, pub_test_df, test_df = prepro1(train_df, pub_test_df, test_df, param)

    train_df = train_df[train_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
    test_df = test_df[test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
    pub_test_df = pub_test_df[pub_test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)

    X = train_df.drop("sig_id", axis=1)
    y_nonv = y_nonv.drop("sig_id", axis=1)
    y_all_nonv = y_all_nonv.drop("sig_id", axis=1).values
    pub_test_X = pub_test_df.drop("sig_id", axis=1)
    test_X = test_df.drop("sig_id", axis=1)
    X.drop("drug_id", axis=1, inplace=True) #

    train_preds = np.zeros((X.shape[0], y_nonv.shape[1]))
    preds = np.zeros((test_X.shape[0], y_nonv.shape[1]))
    
    for seed in seeds:
        oof_preds, preds_seed = train_by_seed1(X, y_nonv, y_all_nonv, pub_test_X, test_X, scored, seed, param, mode)
        train_preds += oof_preds
        preds += preds_seed
    train_preds /= len(seeds)
    preds /= len(seeds)
    return train_preds, preds

def get_ratio_labels(df):
    columns = list(df.columns)
    columns.pop(0)
    ratios = []
    toremove = []
    for c in columns:
        counts = df[c].value_counts()
        if len(counts) != 1:
            ratios.append(counts[0]/counts[1])
        else:
            toremove.append(c)
    print(f"remove {len(toremove)} columns")
    
    for t in toremove:
        columns.remove(t)
    return columns, np.array(ratios).astype(np.int32)


def train_by_fold2(X, y_nonv, y_all_nonv, pub_test_X, test_X, train_index, valid_index, seed, param, fold, mode):
    train_X = X.iloc[train_index]
    train_y = y_nonv[train_index]
    train_y_all = y_all_nonv[train_index]
    valid_X = X.iloc[valid_index]
    valid_y = y_nonv[valid_index]
    valid_y_all = y_all_nonv[valid_index]

    # prepare data for training
    train_X = train_X.values
    valid_X = valid_X.values
    test_X = test_X.values
    print(train_X.shape)

    val_preds, test_preds = run_model2(train_X, train_y, train_y_all, valid_X, valid_y, valid_y_all, test_X, param, seed, fold, mode)
    return val_preds, test_preds

def train_by_seed2(X, y_nonv, y_all_nonv, pub_test_X, test_X, scored, seed, param, mode):
    seed_everything(seed)
    K = param["K"]
    train_preds = np.zeros((X.shape[0], y_nonv.shape[1]))
    preds_seed = np.zeros((test_X.shape[0], y_nonv.shape[1]))
    
    targets = [col for col in y_nonv.columns if col != "sig_id" and col != "drug_id"]
    y_nonv = y_nonv.values
    mskf = MultilabelStratifiedKFold(n_splits=K, random_state=seed, shuffle=True)

    for fold, (tr_idx, val_idx) in enumerate(mskf.split(X, y_nonv)):
        val_preds, preds = train_by_fold2(X, y_nonv, y_all_nonv, pub_test_X, test_X, tr_idx, val_idx, seed, param, fold, mode)
        train_preds[val_idx] = val_preds
        preds_seed += preds
    preds_seed /= K
    print("cv score : {}".format(metric(y_nonv, train_preds)))
    return train_preds, preds_seed


def run2(train_df, pub_test_df, test_df, y_nonv, y_non, y_all_nonv, scored, seeds, param, mode):
    train_df, pub_test_df, test_df = prepro2(train_df, pub_test_df, test_df, param)

    y_non_cols, ratios_nonscored = get_ratio_labels(y_non)

    train_df = train_df[train_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
    test_df = test_df[test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
    pub_test_df = pub_test_df[pub_test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)

    y_all_nonv = y_nonv.merge(y_non[y_non_cols+["sig_id"]], on="sig_id", how="left")

    X = train_df.drop("sig_id", axis=1)
    y_nonv = y_nonv.drop("sig_id", axis=1)
    y_all_nonv = y_all_nonv.drop("sig_id", axis=1).values
    pub_test_X = pub_test_df.drop("sig_id", axis=1)
    test_X = test_df.drop("sig_id", axis=1)
    X.drop("drug_id", axis=1, inplace=True) #

    train_preds = np.zeros((X.shape[0], y_nonv.shape[1]))
    preds = np.zeros((test_X.shape[0], y_nonv.shape[1]))
    
    for seed in seeds:
        oof_preds, preds_seed = train_by_seed2(X, y_nonv, y_all_nonv, pub_test_X, test_X, scored, seed, param, mode)
        train_preds += oof_preds
        preds += preds_seed
    train_preds /= len(seeds)
    preds /= len(seeds)
    return train_preds, preds


def run3(train_df, pub_test_df, test_df, y_nonv, y_all_nonv, scored, seeds, param, mode):
    train_df = train_df[train_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
    test_df = test_df[test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)
    pub_test_df = pub_test_df[pub_test_df["cp_type"] != "ctl_vehicle"].drop("cp_type", axis=1).reset_index(drop=True)

    train_df, pub_test_df, test_df = prepro3(train_df, pub_test_df, test_df, param)

    X = train_df.drop("sig_id", axis=1)
    y_nonv = y_nonv.drop("sig_id", axis=1)
    y_all_nonv = y_all_nonv.drop("sig_id", axis=1).values
    pub_test_X = pub_test_df.drop("sig_id", axis=1)
    test_X = test_df.drop("sig_id", axis=1)
    X.drop("drug_id", axis=1, inplace=True) #

    train_preds = np.zeros((X.shape[0], y_nonv.shape[1]))
    preds = np.zeros((test_X.shape[0], y_nonv.shape[1]))
    
    for seed in seeds:
        oof_preds, preds_seed = train_by_seed1(X, y_nonv, y_all_nonv, pub_test_X, test_X, scored, seed, param, mode)
        train_preds += oof_preds
        preds += preds_seed
    train_preds /= len(seeds)
    preds /= len(seeds)
    return train_preds, preds