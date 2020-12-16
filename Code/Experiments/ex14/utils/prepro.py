import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from scipy.spatial import distance


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


def onehot_timedose(df1, df2, df3):
    drop_cols = ["cp_time", "cp_dose", "time_dose"]
    df1 = pd.concat([pd.get_dummies(df1["time_dose"], prefix="onehot", drop_first=True), df1.drop(drop_cols, axis=1) ], axis=1)
    df2 = pd.concat([pd.get_dummies(df2["time_dose"], prefix="onehot", drop_first=True), df2.drop(drop_cols, axis=1) ], axis=1)
    df3 = pd.concat([pd.get_dummies(df3["time_dose"], prefix="onehot", drop_first=True), df3.drop(drop_cols, axis=1) ], axis=1)
    return df1, df2, df3




def scale_col1(df1, df2, df3, SCALE, seed):
    print("scale")
    cols = [col for col in df1.columns if col.startswith("g-") or col.startswith("c-")]
    scaler = make_scaler(SCALE, seed).fit(df1.append(df2)[cols])
    for df in [df1, df2, df3]:
        df[cols] = scaler.transform(df[cols])
    return df1, df2, df3


def pca1(df1, df2, df3, seed, decomp_g, decomp_c):
    print("PCA")
    decom_g_cols = [f"pca_g-{i}" for i in range(decomp_g)]
    decom_c_cols = [f"pca_c-{i}" for i in range(decomp_c)]
    GENES = [col for col in df1.columns if col.startswith("g-")]
    CELLS = [col for col in df1.columns if col.startswith("c-")]
    pca_g = PCA(n_components = decomp_g, random_state = seed).fit(df1.append(df2)[GENES])
    pca_c = PCA(n_components = decomp_c, random_state = seed).fit(df1.append(df2)[CELLS])

    for df in [df1, df2, df3]:
        df[decom_g_cols] = pca_g.transform(df[GENES])
        df[decom_c_cols] = pca_c.transform(df[CELLS])

    return df1, df2, df3


def VT1(df1, df2, df3, vt):
    print("VT")
    cols = [col for col in df1.columns if col.startswith("c-") or col.startswith("g-") or col.startswith("pca")]
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

def Kmeans1(df1, df2, df3, param):
    print("kmeans")
    GENES = [col for col in df1.columns if col.startswith("g-")]
    CELLS = [col for col in df1.columns if col.startswith("c-")]

    kmeans_g = KMeans(n_clusters=param["kmeans_g"], random_state=100).fit(df1.append(df2)[GENES])  
    for df in [df1, df2, df3]:
        df["kmeans"] = kmeans_g.predict(df[GENES])
    df1 = pd.concat([df1.drop("kmeans", axis=1), pd.get_dummies(df1["kmeans"], prefix="kmeans-g")], axis=1)
    df2 = pd.concat([df2.drop("kmeans", axis=1), pd.get_dummies(df2["kmeans"], prefix="kmeans-g")], axis=1)
    df3 = pd.concat([df3.drop("kmeans", axis=1), pd.get_dummies(df3["kmeans"], prefix="kmeans-g")], axis=1)

    kmeans_c = KMeans(n_clusters=param["kmeans_c"], random_state=100).fit(df1.append(df2)[CELLS])  
    for df in [df1, df2, df3]:
        df["kmeans"] = kmeans_c.predict(df[CELLS])
    df1 = pd.concat([df1.drop("kmeans", axis=1), pd.get_dummies(df1["kmeans"], prefix="kmeans-c-")], axis=1)
    df2 = pd.concat([df2.drop("kmeans", axis=1), pd.get_dummies(df2["kmeans"], prefix="kmeans-c-")], axis=1)
    df3 = pd.concat([df3.drop("kmeans", axis=1), pd.get_dummies(df3["kmeans"], prefix="kmeans-c-")], axis=1)

    return df1, df2, df3

def KmeansPCA1(df1, df2, df3, param):
    PCAS = [col for col in df1.columns if col.startswith("pca")]
    kmeans_pca = KMeans(n_clusters=param["kmeans_pca"], random_state=100).fit(df1.append(df2)[PCAS])  
    for df in [df1, df2, df3]:
        df["kmeans"] = kmeans_pca.predict(df[PCAS])
    df1 = pd.concat([df1.drop("kmeans", axis=1), pd.get_dummies(df1["kmeans"], prefix="kmeans-pca")], axis=1)
    df2 = pd.concat([df2.drop("kmeans", axis=1), pd.get_dummies(df2["kmeans"], prefix="kmeans-pca")], axis=1)
    df3 = pd.concat([df3.drop("kmeans", axis=1), pd.get_dummies(df3["kmeans"], prefix="kmeans-pca")], axis=1)

    return df1, df2, df3

def stats1(df1, df2, df3):
    print("stats")
    GENES = [col for col in df1.columns if col.startswith("g-")]
    CELLS = [col for col in df1.columns if col.startswith("c-")]
    BIOS = GENES + CELLS
    gsquarecols=['g-574','g-211','g-216','g-0','g-255','g-577','g-153','g-389','g-60','g-370','g-248','g-167','g-203','g-177','g-301','g-332','g-517','g-6','g-744','g-224','g-162','g-3','g-736','g-486','g-283','g-22','g-359','g-361','g-440','g-335','g-106','g-307','g-745','g-146','g-416','g-298','g-666','g-91','g-17','g-549','g-145','g-157','g-768','g-568','g-396']

    for df in [df1, df2, df3]:    
        df['g_sum'] = df[GENES].sum(axis = 1)
        df['g_mean'] = df[GENES].mean(axis = 1)
        df['g_std'] = df[GENES].std(axis = 1)
        df['g_kurt'] = df[GENES].kurtosis(axis = 1)
        df['g_skew'] = df[GENES].skew(axis = 1)
        df['c_sum'] = df[CELLS].sum(axis = 1)
        df['c_mean'] = df[CELLS].mean(axis = 1)
        df['c_std'] = df[CELLS].std(axis = 1)
        df['c_kurt'] = df[CELLS].kurtosis(axis = 1)
        df['c_skew'] = df[CELLS].skew(axis = 1)
        df['gc_sum'] = df[BIOS].sum(axis = 1)
        df['gc_mean'] = df[BIOS].mean(axis = 1)
        df['gc_std'] = df[BIOS].std(axis = 1)
        df['gc_kurt'] = df[BIOS].kurtosis(axis = 1)
        df['gc_skew'] = df[BIOS].skew(axis = 1)
        
        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-23'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']

        for feature in CELLS:
            df[f'{feature}_squared'] = df[feature] ** 2     
                
        for feature in gsquarecols:
            df[f'{feature}_squared'] = df[feature] ** 2 

    return df1, df2, df3


def prepro1(df1, df2, df3, param):
    df1, df2, df3 = time_dose(df1, df2, df3)
    df1, df2, df3 = onehot_timedose(df1, df2, df3)
    df1, df2, df3 = Kmeans1(df1, df2, df3, param)
    df1, df2, df3 = scale_col1(df1, df2, df3, param["SCALE"], 100)
    df1, df2, df3 = pca1(df1, df2, df3, 100, param["decomp_g"], param["decomp_c"])
    df1, df2, df3 = VT1(df1, df2, df3, param["VT"])
    df1, df2, df3 = KmeansPCA1(df1, df2, df3, param)
    df1, df2, df3 = stats1(df1, df2, df3)

    return df1, df2, df3




def prepro2(df1, df2, df3, param):
    df1, df2, df3 = scale_col1(df1, df2, df3, param["SCALE"], 101)
    df1, df2, df3 = pca1(df1, df2, df3, 101, param["decomp_g"], param["decomp_c"])
    ådf1, df2, df3 = VT1(df1, df2, df3, param["VT"])
    df1, df2, df3 = time_dose(df1, df2, df3)
    df1, df2, df3 = onehot_timedose(df1, df2, df3)
    return df1, df2, df3, 


def stats3(df1, df2, df3):
    
    features_g = [col for col in df1.columns if col.startswith("g-")]
    features_c = [col for col in df1.columns if col.startswith("c-")]
    
    for df in [df1, df2, df3]:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return df1, df2, df3

def c_squared3(df1, df2, df3):
    
    features_c = [col for col in df1.columns if col.startswith("c-")]
    for df in [df1, df2, df3]:
        for feature in features_c:
            df[f'{feature}_squared'] = df[feature] ** 2
    return df1, df2, df3


def prepro3(df1, df2, df3, param):
    df1, df2, df3 = time_dose(df1, df2, df3)
    df1, df2, df3 = onehot_timedose(df1, df2, df3)
    df1, df2, df3 = stats3(df1, df2, df3)
    df1, df2, df3 = c_squared3(df1, df2, df3)
    df1, df2, df3 = pca1(df1, df2, df3, 102, param["decomp_g"], param["decomp_c"])
    df1, df2, df3 = scale_col1(df1, df2, df3, param["SCALE"], 102)
    return df1, df2, df3
