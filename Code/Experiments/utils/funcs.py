import warnings, random, os, sys, tqdm, time
sys.path.append("../")
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold
from scipy.spatial import distance
from scipy.stats import pearsonr

import torch
from .torch_utils import run_dnn1


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
        
    
def make_scaler(flag, seed):
    if flag == "quantile":
        return QuantileTransformer(n_quantiles=100,random_state=seed, output_distribution="normal")
    elif flag == "standard":
        return StandardScaler()
    elif flag == "minmax":
        return MinMaxScaler()
    elif flag == "robust":
        return RobustScaler()


def time_dose(df1, df2, df3):
    for df in [df1, df2, df3]:
        df["time_dose"] = df["cp_time"].astype(str) + " * " + df["cp_dose"]
    
    return df1, df2, df3


def prod(df1, df2, df3):
    prod_cols = [['g-145', 'g-201', 'g-208'], ['g-370', 'g-508', 'g-37'], ['g-38', 'g-392', 'g-707'], ['g-328', 'g-28', 'g-392'], ['g-441', 'g-157', 'g-392'], ['g-181', 'g-100', 'g-392'], ['g-67', 'g-760', 'g-50'], ['g-731', 'g-100', 'g-707'], ['g-478', 'g-468', 'g-310'], ['g-91', 'g-145', 'g-208'], ['g-106', 'g-744', 'g-91'], ['g-131', 'g-208', 'g-392'], ['g-144', 'g-123', 'g-86'], ['g-228', 'g-72', 'g-67'], ['g-31', 'g-328', 'g-460'], ['g-392', 'g-731', 'g-100'], ['g-732', 'g-744', 'g-707'], ['g-705', 'g-375', 'g-704'], ['g-508', 'g-50', 'g-411'], ['g-234', 'g-58', 'g-520'], ['g-503', 'g-761', 'g-50'], ['g-113', 'g-75', 'g-178'], ['g-50', 'g-508', 'g-113'], ['g-113', 'g-375', 'g-75'], ['g-576', 'g-452', 'g-392'], ['g-50', 'g-37', 'g-36'], ['g-707', 'g-133', 'g-392'], ['g-484', 'g-392', 'g-544'], ['g-508', 'g-67', 'g-370'], ['g-123', 'g-731', 'g-100'], ['g-298', 'g-477', 'g-644'], ['g-72', 'g-370', 'g-50'], ['g-67', 'g-178', 'g-113'], ['g-744', 'g-608', 'g-100'], ['g-91', 'g-100', 'g-707'], ['g-37', 'g-228', 'g-202'], ['g-37', 'g-300', 'g-370'], ['g-234', 'g-508', 'g-595'], ['g-596', 'g-744', 'g-707'], ['g-300', 'g-227', 'g-591'], ['g-135', 'g-392', 'g-512'], ['g-731', 'g-744', 'g-158'], ['g-69', 'g-707', 'g-100'], ['g-276', 'g-653', 'g-291'], ['g-624', 'g-615', 'g-189'], ['g-181', 'g-707', 'g-38'], ['g-72', 'g-75', 'g-508'], ['g-231', 'g-707', 'g-392'], ['g-508', 'g-37', 'g-72'], ['g-725', 'g-712', 'g-640'], ['g-67', 'g-644', 'g-113'], ['g-508', 'g-228', 'g-656'], ['g-185', 'g-37', 'g-672'], ['g-370', 'g-50', 'g-503'], ['g-201', 'g-745', 'g-599'], ['g-332', 'g-50', 'g-571'], ['g-50', 'g-37', 'g-59'], ['g-508', 'g-113', 'g-231'], ['g-707', 'g-158', 'g-100'], ['g-257', 'g-50', 'g-72']]
    for cols in prod_cols:
        name = "prod-" + " * ".join(cols)
        df1[name] = df1[cols].mean(axis=1)
        df2[name] = df2[cols].mean(axis=1)
        df3[name] = df3[cols].mean(axis=1)
    return df1, df2, df3


def VT(df1, df2, df3, vt):
    cols = [col for col in df1.columns if col.startswith("c-") or col.startswith("g-") or col.startswith("prod-")]
    drop_cols = []
    temp = pd.concat([df1, df2])
    for col in cols:
        if temp[col].var() <= vt:
            drop_cols.append(col)
    df1.drop(columns=drop_cols, inplace=True)
    df2.drop(columns=drop_cols, inplace=True)
    df3.drop(columns=drop_cols, inplace=True)
    print("drop cols num : {}".format(len(drop_cols)))
    return df1, df2, df3

def onehot_timedose(df1, df2, df3):
    drop_cols = ["cp_time", "cp_dose", "time_dose"]
    df1 = pd.concat([pd.get_dummies(df1["time_dose"], prefix="onehot", drop_first=True), df1.drop(drop_cols, axis=1) ], axis=1)
    df2 = pd.concat([pd.get_dummies(df2["time_dose"], prefix="onehot", drop_first=True), df2.drop(drop_cols, axis=1) ], axis=1)
    df3 = pd.concat([pd.get_dummies(df3["time_dose"], prefix="onehot", drop_first=True), df3.drop(drop_cols, axis=1) ], axis=1)
    return df1, df2, df3

def stats(df1, df2, df3):
    GENES = [col for col in df1.columns if col.startswith("g-")]
    CELLS = [col for col in df1.columns if col.startswith("c-")]
    BIOS = GENES + CELLS
    for df in [df1, df2, df3]:
        df["agg-sum-g"] = df[GENES].sum(axis=1)
        df["agg-mean-g"] = df[GENES].mean(axis=1)
        df["agg-std-g"] = df[GENES].std(axis=1)
        df["agg-kurt-g"] = df[GENES].kurt(axis=1)
        df["agg-skew-g"] = df[GENES].skew(axis=1)
        df["agg-sum-c"] = df[CELLS].sum(axis=1)
        df["agg-mean-c"] = df[CELLS].mean(axis=1)
        df["agg-std-c"] = df[CELLS].std(axis=1)
        df["agg-kurt-c"] = df[CELLS].kurt(axis=1)
        df["agg-skew-c"] = df[CELLS].skew(axis=1)
        df["agg-sum-gc"] = df[BIOS].sum(axis=1)
        df["agg-mean-gc"] = df[BIOS].mean(axis=1)
        df["agg-std-gc"] = df[BIOS].std(axis=1)
        df["agg-kurt-gc"] = df[BIOS].kurt(axis=1)
        df["agg-skew-gc"] = df[BIOS].skew(axis=1)
    return df1, df2, df3

def prepro1(df1, df2, df3, vt):
    print("prepro1")
    # df1 : train
    # df2 : pubtest
    # df3 : test
    df1, df2, df3 = time_dose(df1, df2, df3)
    df1, df2, df3 = prod(df1, df2, df3)
    df1, df2, df3 = VT(df1, df2, df3, vt)
    df1, df2, df3 = onehot_timedose(df1, df2, df3)
    df1, df2, df3 = stats(df1, df2, df3)

    return df1, df2, df3


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


def scale_col(X1, X2, X3, X4, SCALE, seed):
    cols = [col for col in X1.columns if col.startswith("g-") or col.startswith("c-")]
    scaler = make_scaler(SCALE, seed).fit(X1.append(X3)[cols])
    for df in [X1, X2, X3, X4]:
        df[cols] = scaler.transform(df[cols])
    return X1, X2, X3, X4


def make_pca(X1, X2, X3, X4, seed, decomp_g, decomp_c):
    print("PCA")
    decom_g_cols = [f"pca_g-{i}" for i in range(decomp_g)]
    decom_c_cols = [f"pca_c-{i}" for i in range(decomp_c)]
    GENES = [col for col in X1.columns if col.startswith("g-")]
    CELLS = [col for col in X1.columns if col.startswith("c-")]
    pca_g = PCA(n_components = decomp_g, random_state = seed).fit(X1.append(X3)[GENES])
    pca_c = PCA(n_components = decomp_c, random_state = seed).fit(X1.append(X3)[CELLS])

    for df in [X1, X2, X3, X4]:
        df[decom_g_cols] = pca_g.transform(df[GENES])
        df[decom_c_cols] = pca_c.transform(df[CELLS])

    return X1, X2, X3, X4


def prepro_infold(X1, X2, X3, X4, SCALE, seed, param):
    """
    X1 : train_X
    X2 : valid_X
    X3 : pub_test_X
    X4 : test_X
    """
    X1, X2, X3, X4 = scale_col(X1, X2, X3, X4, SCALE, seed)
    X1, X2, X3, X4 = make_pca(X1, X2, X3, X4, seed, param["decomp_g"], param["decomp_c"])
    return X1, X2, X3, X4

def train_by_fold(X, y_nonv, pub_test_X, test_X, seed, SCALE, param, scored, fold, mode):
    train_index = scored[scored["fold"] != fold].index.to_list()
    valid_index = scored[scored["fold"] == fold].index.to_list()

    train_X = X.iloc[train_index]
    train_y = y_nonv[train_index]
    valid_X = X.iloc[valid_index]
    valid_y = y_nonv[valid_index]

    train_X, valid_X, pub_test_X, test_X = prepro_infold(train_X, valid_X, pub_test_X, test_X, SCALE, seed, param)

    # prepare data for training
    train_X = train_X.values
    valid_X = valid_X.values
    test_X = test_X.values
    print(train_X.shape)

    val_preds, test_preds = run_dnn1(train_X, train_y, valid_X, valid_y, test_X, param, seed, fold, mode)
    return valid_index, val_preds, test_preds


def train_by_seed(X, y_nonv, pub_test_X, test_X, scored, seed, SCALE, param, mode):
    seed_everything(seed)
    K = param["K"]
    train_preds = np.zeros((X.shape[0], y_nonv.shape[1]))
    preds_seed = np.zeros((test_X.shape[0], y_nonv.shape[1]))
    
    targets = [col for col in y_nonv.columns if col != "sig_id" and col != "drug_id"]
    y_nonv = y_nonv.values
    scored = make_folds(targets, scored, seed, K)

    for fold in range(K):
        val_index, val_preds, preds = train_by_fold(X, y_nonv, pub_test_X, test_X, seed, SCALE, param, scored, fold, mode)
        train_preds[val_index] = val_preds
        preds_seed += preds
    preds_seed /= K
    print("cv score : {}".format(metric(y_nonv, train_preds)))
    return train_preds, preds_seed



def run(train_df, pub_test_df, test_df, y_nonv, scored, seeds, SCALE, param, mode):
    train_df, pub_test_df, test_df = prepro1(train_df, pub_test_df, test_df, param["VT"])

    X = train_df.drop("sig_id", axis=1)
    y_nonv = y_nonv.drop("sig_id", axis=1)
    pub_test_X = pub_test_df.drop("sig_id", axis=1)
    test_X = test_df.drop("sig_id", axis=1)
    print(X.shape)
    X.drop("drug_id", axis=1, inplace=True) #

    train_preds = np.zeros((X.shape[0], y_nonv.shape[1]))
    preds = np.zeros((test_X.shape[0], y_nonv.shape[1]))
    
    for seed in seeds:
        oof_preds, preds_seed = train_by_seed(X, y_nonv, pub_test_X, test_X, scored, seed, SCALE, param, mode)
        train_preds += oof_preds
        preds += preds_seed
    train_preds /= len(seeds)
    preds /= len(seeds)
    return train_preds, preds