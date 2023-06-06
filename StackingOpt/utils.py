import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import lightgbm as lgb
import torch
import random
from dateutil import relativedelta

### 良く使う
#　フォルダー内のファイルバス一覧を取得(EDA2001)
def get_file_paths(folder_path):
    file_paths = []
    for root, directories, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths

# seedの固定
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

###　データの前処理
##EDA003より

#元データを取得
def get_preprocessing_data(to_unique_id,folder_path = "E:/study/preprocessing_data/1_twoweeks_nv/"):
    df_list = []
    for id in to_unique_id:
        #path = f"E:/study/preprocessing_data/1_twoweeks_nv/{id}.csv"
        path = folder_path + f"{id}.csv"
        df = pd.read_csv(path, header=None, index_col=None)
        df.columns = ["year","month","day","hour","flag","nv","two_weeks_max"]
        df["id"] = int(id)
        df_list.append(df)
        
    df = pd.concat(df_list) 
    df["year"] += 2000
    df["minute"] = df["hour"]%1*60
    df["hour"] = df["hour"]//1
    df[["hour","minute"]] = df[["hour","minute"]].astype(int)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])


    df["month_angle"] = (df["month"] - 1) * (2 * np.pi / 12)  # 月は1~12の範囲を持つので、1を引く
    df["day_angle"] = (df["day"] - 1) * (2 * np.pi / 30)  # 仮に各月を30日として計算
    df["hour_angle"] = df["hour"] * (2 * np.pi / 24)

    df["month_sin"] = np.sin(df["month_angle"])
    df["month_cos"] = np.cos(df["month_angle"])
    df["day_sin"] = np.sin(df["day_angle"])
    df["day_cos"] = np.cos(df["day_angle"])
    df["hour_sin"] = np.sin(df["hour_angle"])
    df["hour_cos"] = np.cos(df["hour_angle"])

    base_datetime = datetime(1970, 1, 1)
    df["year_seconds"] = (df["datetime"] - base_datetime).dt.total_seconds()
    seconds_in_a_year = (365.25 * 24 * 60 * 60)  
    df["year_angle"] = df["year_seconds"] * (2 * np.pi / seconds_in_a_year)
    df["year_sin"] = np.sin(df["year_angle"])
    df["year_cos"] = np.cos(df["year_angle"])

    df["generation"] = df["nv"]*df["two_weeks_max"]
    return df

#↑の簡略版、元データを一つづつ取得(EDA2001)
def get_preprocessing_data2(id):

    path = f"E:/study/preprocessing_data/1_twoweeks_nv/{id}.csv"
    df = pd.read_csv(path, header=None, index_col=None)
    df.columns = ["year","month","day","hour","flag","nv","two_weeks_max"]
    df["id"] = int(id)

    df["year"] += 2000
    df["minute"] = df["hour"]%1*60
    df["hour"] = df["hour"]//1
    df[["hour","minute"]] = df[["hour","minute"]].astype(int)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

    df["generation"] = df["nv"]*df["two_weeks_max"]
    return df

# ↑↑でフォルダーのpathを変更している。(EDA2001)
def get_preprocessing_data3(to_unique_id,folder_path = 'E:/study/output/StackingOpt/EDA2001/'):
    df_list = []
    for id in to_unique_id:
        #path = f"E:/study/preprocessing_data/1_twoweeks_nv/{id}.csv"
        path = folder_path + f"{id}_201308150000_201408010000.csv"
        df = pd.read_csv(path, index_col=None)
        df["id"] = int(id)
        df_list.append(df)
        
    df = pd.concat(df_list) 

    df['datetime'] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute


    df["month_angle"] = (df["month"] - 1) * (2 * np.pi / 12)  # 月は1~12の範囲を持つので、1を引く
    df["day_angle"] = (df["day"] - 1) * (2 * np.pi / 30)  # 仮に各月を30日として計算
    df["hour_angle"] = df["hour"] * (2 * np.pi / 24)

    df["month_sin"] = np.sin(df["month_angle"])
    df["month_cos"] = np.cos(df["month_angle"])
    df["day_sin"] = np.sin(df["day_angle"])
    df["day_cos"] = np.cos(df["day_angle"])
    df["hour_sin"] = np.sin(df["hour_angle"])
    df["hour_cos"] = np.cos(df["hour_angle"])

    base_datetime = datetime(1970, 1, 1)
    df["year_seconds"] = (df["datetime"] - base_datetime).dt.total_seconds()
    seconds_in_a_year = (365.25 * 24 * 60 * 60)  
    df["year_angle"] = df["year_seconds"] * (2 * np.pi / seconds_in_a_year)
    df["year_sin"] = np.sin(df["year_angle"])
    df["year_cos"] = np.cos(df["year_angle"])

    df["generation"] = df["nv"]*df["two_weeks_max"]
    return df

#30分前の発電量を取得(LGBM001)
def prev_30m_generation(group):
    group = group.set_index('datetime')
    group = group.resample('30T').asfreq().reset_index()
    group['prev_30m_generation'] = group['generation'].shift(1)
    return group

'''
import utils

unique_id = ['10000095', '10000269', '1020000002', '1110000001', '1110000010', '1110000011', '1110000012', '1110000013', '1110000014', '1110000015', '1160000025', '1160000090', '1160000091', '1160000182', '1160000185', '1160000253', '1160000387', '1160000402', '1160000419', '1160000420', '1160000423', '1270000026', '1280000048', '1550000001', '1650000004', '1680000001', '1680000002', '1680000003', '1680000004', '1680000010', '1680000017', '1680000021', '1680000033', '1680000047', '1680000054', '1680000057', '1680000063', '1680000067', '1680000080', '1680000081', '1680000097', '1680000107', '1680000108', '1680000112', '1680000151', '1680000152', '1680000213', '1680000216', '1680000217', '1680000218', '1680000223', '1680000228', '1680000285', '1680000287', '1680000327', '1680000364', '2220000001', '2220000002', '2220000003', '2730000001', '2910000002', '3000000007', '3000000012', '3000000042', '5000000044', '5000000045', '6000000016', '6000000017', '6060000016', '6060000017', '6060000018', '6170000016', '6170000123', '6170000124', '6170000125', '6620000065', '6620000066', '6620000088', '6620000089', '6620000111', '6620000117', '6620000118', '6620000121', '6620000122', '6620000123', '6620000124', '6620000131', '6620000132', '6910000180', '6910000198', '6910000200', '6910000206', '6910000216', '6910000217', '6910000239', '6910000240', '6910000249', '6910000250', '6910000421', '6910000424', '6910000425', '6910000469', '6910000470']
unique_id = [int(i) for i in unique_id]
to_unique_id = [str(num).zfill(10) for num in unique_id]
df = utils.get_preprocessing_data(to_unique_id)

df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=["id","datetime"],inplace=True)


'''

# #オプティカルフローの予測値を抽出,EDA005で改良
# def get_prediction_data(id_data,lat,lon,dt,folder_path = "E:/study/prediction_data/parameter_lambda_0.019_iterate_1000_p3/env/"):
    
#     path = folder_path + f"{dt.month}月/{dt.month}月{dt.day}日/NV{dt.year%1000}{dt.month}{dt.day}{dt.hour}.{int(dt.minute/60*10)}_0.02_0.019_1000_p3_env.csv"
#     df = pd.read_csv(path, header=None, index_col=None)

#     min_lat,max_lat = id_data.id_lat.min(),id_data.id_lat.max()
#     min_lng,max_lng = id_data.id_lng.min(),id_data.id_lng.max()
#     userow = (lon.iloc[:,0]>=min_lng)&(lon.iloc[:,0]<=max_lng)
#     usecol = (lat.iloc[0]>=min_lat)&(lat.iloc[0]<=max_lat)

#     df = df.loc[userow,usecol]
#     df = df.replace({ " nan": np.nan," -nan(ind)":np.nan})
#     df = df.astype(float)

#     pred_data = pd.DataFrame(zip(df.to_numpy().reshape(-1),\
#                             lat.loc[userow,usecol].to_numpy().reshape(-1),\
#                             lon.loc[userow,usecol].to_numpy().reshape(-1)),\
#                             columns=["pred","id_lat_mesh","id_lng_mesh"])
#     pred_data["datetime"] = dt
#     return df,pred_data


'''
lat = pd.read_csv(r"E:\study\preprocessing_data\3_mesh_place\lati_zenkoku.csv", header=None, index_col=None)
lon = pd.read_csv(r"E:\study\preprocessing_data\3_mesh_place\long_zenkoku.csv", header=None, index_col=None)
id_all_data = pd.read_csv(r"E:\study\preprocessing_data\id_all_data.csv", encoding="shift_jis")

unique_id = ['10000095', '10000269', '1020000002', '1110000001', '1110000010', '1110000011', '1110000012', '1110000013', '1110000014', '1110000015', '1160000025', '1160000090', '1160000091', '1160000182', '1160000185', '1160000253', '1160000387', '1160000402', '1160000419', '1160000420', '1160000423', '1270000026', '1280000048', '1550000001', '1650000004', '1680000001', '1680000002', '1680000003', '1680000004', '1680000010', '1680000017', '1680000021', '1680000033', '1680000047', '1680000054', '1680000057', '1680000063', '1680000067', '1680000080', '1680000081', '1680000097', '1680000107', '1680000108', '1680000112', '1680000151', '1680000152', '1680000213', '1680000216', '1680000217', '1680000218', '1680000223', '1680000228', '1680000285', '1680000287', '1680000327', '1680000364', '2220000001', '2220000002', '2220000003', '2730000001', '2910000002', '3000000007', '3000000012', '3000000042', '5000000044', '5000000045', '6000000016', '6000000017', '6060000016', '6060000017', '6060000018', '6170000016', '6170000123', '6170000124', '6170000125', '6620000065', '6620000066', '6620000088', '6620000089', '6620000111', '6620000117', '6620000118', '6620000121', '6620000122', '6620000123', '6620000124', '6620000131', '6620000132', '6910000180', '6910000198', '6910000200', '6910000206', '6910000216', '6910000217', '6910000239', '6910000240', '6910000249', '6910000250', '6910000421', '6910000424', '6910000425', '6910000469', '6910000470']
unique_id = [int(i) for i in unique_id]
id_data = id_all_data[id_all_data.id.isin(unique_id)].reset_index(drop=True)

dt= datetime.datetime(2013, 8, 15, 6, 30)
df,pred_data = utils.get_prediction_data(id_data,lat,lon,dt)
'''



##EDA005

#オプティカルフローの予測値を抽出(EDA005)
def get_prediction_data2(id_data,lat,lon,dt,folder_path = "E:/study/prediction_data/parameter_lambda_0.019_iterate_1000_p3/env/"):
    
    path = folder_path + f"{dt.month}月/{dt.month}月{dt.day}日/NV{dt.year%1000}{dt.month}{dt.day}{dt.hour}.{int(dt.minute/60*10)}_0.02_0.019_1000_p3_env.csv"
    df = pd.read_csv(path, header=None, index_col=None)

    min_lat,max_lat = id_data.id_lat_mesh.min(),id_data.id_lat_mesh.max()
    min_lng,max_lng = id_data.id_lng_mesh.min(),id_data.id_lng_mesh.max()
    userow = (lon.iloc[:,0]>=min_lng)&(lon.iloc[:,0]<=max_lng)
    usecol = (lat.iloc[0]>=min_lat)&(lat.iloc[0]<=max_lat)

    df = df.loc[userow,usecol]
    df = df.replace({ " nan": np.nan," -nan(ind)":np.nan})
    df = df.astype(float)

    pred_data = pd.DataFrame(zip(df.to_numpy().reshape(-1),\
                            lat.loc[userow,usecol].to_numpy().reshape(-1),\
                            lon.loc[userow,usecol].to_numpy().reshape(-1)),\
                            columns=["pred","id_lat_mesh","id_lng_mesh"])
    pred_data["datetime"] = dt
    return df,pred_data

# 全ての予測値を抽出(EDA006)
def get_prediction_data_all(lat,lon,dt,folder_path = "E:/study/prediction_data/parameter_lambda_0.019_iterate_1000_p3/env/"):
    
    path = folder_path + f"{dt.month}月/{dt.month}月{dt.day}日/NV{dt.year%1000}{dt.month}{dt.day}{dt.hour}.{int(dt.minute/60*10)}_0.02_0.019_1000_p3_env.csv"
    df = pd.read_csv(path, header=None, index_col=None)

    df = df.replace({ " nan": np.nan," -nan(ind)":np.nan,"-nan(ind)":np.nan})
    df = df.astype(float)

    pred_data = pd.DataFrame(zip(df.to_numpy().reshape(-1),\
                            lat.to_numpy().reshape(-1),\
                            lon.to_numpy().reshape(-1)),\
                            columns=["pred","id_lat_mesh","id_lng_mesh"])
    pred_data["datetime"] = dt
    return df,pred_data

# フローの速さを取得(EDA006)
def get_flo_mesh(id_data,lat,lon,dt):

    path = f"E:/study/prediction_data/parameter_lambda_0.019_iterate_1000_p3/flo/{dt.month}月/{dt.month}月{dt.day}日/NV{dt.year%1000}{dt.month}{dt.day}{dt.hour}.{int(dt.minute/60*10)}_0.02_0.019_1000_p3_flo.csv"
    flo = pd.read_csv(path, header=None, index_col=None)

    flo_lat = flo[[c for i,c in enumerate(flo.columns) if i%2==0]]
    flo_lon = flo[[c for i,c in enumerate(flo.columns) if i%2==1]]

    flo_lat.columns = list(range(flo_lat.shape[1]))
    flo_lon.columns = list(range(flo_lon.shape[1]))

    min_lat,max_lat = id_data.id_lat_mesh.min(),id_data.id_lat_mesh.max()
    min_lng,max_lng = id_data.id_lng_mesh.min(),id_data.id_lng_mesh.max()

    userow = (lon.iloc[:,0]>=min_lng)&(lon.iloc[:,0]<=max_lng)
    usecol = (lat.iloc[0]>=min_lat)&(lat.iloc[0]<=max_lat)

    flo_lat = flo_lat.loc[userow,usecol]
    flo_lon = flo_lon.loc[userow,usecol]
    flo = (flo_lat**2+flo_lon**2)**(1/2)
    return flo,flo_lat,flo_lon

# 以下3つは↑のフローの速さの絶対値、緯度、経度を取得(EDA006)
def get_flo_mesh_data(flo_dir,date_list):
    X_data = []
    for date in date_list:
        train_csv = os.path.join(flo_dir, f"flo_data_{date}.csv")
        train_data = pd.read_csv(train_csv, index_col=0)
        X_data.append(train_data.values.flatten())
    X_data = np.array(X_data)
    return X_data

def get_flo_lat_mesh_data(flo_dir,date_list):
    X_data = []
    for date in date_list:
        train_csv = os.path.join(flo_dir, f"flo_lat_data_{date}.csv")
        train_data = pd.read_csv(train_csv, index_col=0)
        X_data.append(train_data.values.flatten())
    X_data = np.array(X_data)
    return X_data

def get_flo_lon_mesh_data(flo_dir,date_list):
    X_data = []
    for date in date_list:
        train_csv = os.path.join(flo_dir, f"flo_lon_data_{date}.csv")
        train_data = pd.read_csv(train_csv, index_col=0)
        X_data.append(train_data.values.flatten())
    X_data = np.array(X_data)
    return X_data



# オプティカルフローの予測値を欠損値補完したmeshを抽出(EDA006)
def get_pred_interpolated_mesh(id_data,lat,lon,dt):
    path = f"E:/study/output/StackingOpt/EDA006/{dt.month}月/{dt.month}月{dt.day}日/Pred_interpolated_{dt.year%1000}{dt.month}{dt.day}{dt.hour}.{int(dt.minute/60*10)}0.02_int2.csv"
    df = pd.read_csv(path, header=None, index_col=None)

    min_lat,max_lat = id_data.id_lat_mesh.min(),id_data.id_lat_mesh.max()
    min_lng,max_lng = id_data.id_lng_mesh.min(),id_data.id_lng_mesh.max()

    userow = (lon.iloc[:,0]>=min_lng)&(lon.iloc[:,0]<=max_lng)
    usecol = (lat.iloc[0]>=min_lat)&(lat.iloc[0]<=max_lat)
    df = df.loc[userow,usecol]
    return df

#↑で作ったオプ補間データを抽出(EDA006)
def get_pred_interpolated_mesh_data(pred_dir,date_list):
    X_data = []
    for date in date_list:
        train_csv = os.path.join(pred_dir, f"pred_interpolated_mesh_data_{date}.csv")
        train_data = pd.read_csv(train_csv, index_col=0)
        X_data.append(train_data.values.flatten())
    X_data = np.array(X_data)
    return X_data

# 欠損値補間のデータを抽出(EDA005)
def get_interpolated_mesh(id_data,lat,lon,dt):
    path = f"E:/study/preprocessing_data/4_interpolated_mesh/{dt.month}月/{dt.month}月{dt.day}日/NV{dt.year%1000}{dt.month}{dt.day}{dt.hour}.{int(dt.minute/60*10)}0.02_int.csv"
    df = pd.read_csv(path, header=None, index_col=None)

    min_lat,max_lat = id_data.id_lat_mesh.min(),id_data.id_lat_mesh.max()
    min_lng,max_lng = id_data.id_lng_mesh.min(),id_data.id_lng_mesh.max()

    userow = (lon.iloc[:,0]>=min_lng)&(lon.iloc[:,0]<=max_lng)
    usecol = (lat.iloc[0]>=min_lat)&(lat.iloc[0]<=max_lat)
    df = df.loc[userow,usecol]
    return df

# ↑の欠損値補完データを抽出(EDA006)
def get_interpolated_mesh_data(interpolated_dir,date_list):
    X_data,Y_data = [],[]
    for date in date_list:
        target_csv = os.path.join(interpolated_dir, f"interpolated_mesh_data_{date}.csv")
        target_data = pd.read_csv(target_csv, index_col=0)
        Y_data.append(target_data.values.flatten())
         
        date2 = datetime.strptime(date, '%Y%m%d%H%M')
        date2 = date2 - timedelta(minutes=30)
        date2 = date2.strftime('%Y%m%d%H%M')
        train_csv = os.path.join(interpolated_dir, f"interpolated_mesh_data_{date2}.csv")
        train_data = pd.read_csv(train_csv, index_col=0)
        X_data.append(train_data.values.flatten())
    Y_data = np.array(Y_data)
    X_data = np.array(X_data)
    return X_data,Y_data

# unique_idのfloの予測値を抽出(LGBM004.ipynb)
def get_unique_pred_interpolated(flo_unique_dir,date_list,unique_id):
    X_data = []
    for date in date_list:
        train_csv = os.path.join(flo_unique_dir, f"pred_id_all_data_{date}.csv")
        train_data = pd.read_csv(train_csv)
        train_data = train_data[train_data.id.isin(unique_id)].sort_values(by="id")
        X_data.append(train_data["pred_interpolated"].values.flatten())
    X_data = np.array(X_data)
    return X_data

#id_lat_mesh,id_lng_meshを小数点第2位にする。(EDA005)
def get_mesh_round(df):
    df["id_lat_mesh"] = df["id_lat_mesh"].round(2)
    df["id_lng_mesh"] = df["id_lng_mesh"].round(2)
    return df


## 特徴量生成
def make_features1(df):
    df["datetime"] = pd.to_datetime(df["datetime"])

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute

    df["month_angle"] = (df["month"] - 1) * (2 * np.pi / 12)  # 月は1~12の範囲を持つので、1を引く
    df["day_angle"] = (df["day"] - 1) * (2 * np.pi / 30)  # 仮に各月を30日として計算
    df["hour_angle"] = df["hour"] * (2 * np.pi / 24)

    df["month_sin"] = np.sin(df["month_angle"])
    df["month_cos"] = np.cos(df["month_angle"])
    df["day_sin"] = np.sin(df["day_angle"])
    df["day_cos"] = np.cos(df["day_angle"])
    df["hour_sin"] = np.sin(df["hour_angle"])
    df["hour_cos"] = np.cos(df["hour_angle"])

    base_datetime = datetime(1970, 1, 1)
    df["year_seconds"] = (df["datetime"] - base_datetime).dt.total_seconds()
    seconds_in_a_year = (365.25 * 24 * 60 * 60)  
    df["year_angle"] = df["year_seconds"] * (2 * np.pi / seconds_in_a_year)
    df["year_sin"] = np.sin(df["year_angle"])
    df["year_cos"] = np.cos(df["year_angle"])

    df["generation"] = df["nv"]*df["two_weeks_max"]
    return df

## データの読み込み
# 使用できるPV4885台のデータ(EDA2002,2分くらい)
def load_data1():
    id_all_data = pd.read_csv("E:\study\output\StackingOpt\EDA006\id_all_data.csv")

    df = pd.read_csv('E:/study/output/StackingOpt/EDA2002/pv_all.csv')
    df.dropna(subset=["prev_30m_generation"],inplace=True)
    df = make_features1(df)

    df = df.merge(id_all_data,on=["id"],how="left")
    df = df[df["observed_max2"]!=0]  # df["observed_max2"]==0のデータを省く。
    df["nv2"] = df["generation"] / df["observed_max2"]

    unique_id = df.id.unique().tolist()
    return df,unique_id

# 実測値、予測値欠損無し390台(LGBM2002.py)
def load_data2():
    id_all_data = pd.read_csv("E:\study\output\StackingOpt\EDA006\id_all_data.csv")

    unique_id = [6240000001, 1670000003, 6310000002, 6310000001, 6900000114, 6900000110, 6900000109, 6900000108, 6900000105, 6900000104, 6900000094, 6900000093, 6900000090, 2100000086, 2100000085, 2100000084, 6900000081, 6900000080, 2100000068, 2100000067, 2100000066, 6910000449, 6910000448, 6900000131, 6950000001, 6900000173, 6900000134, 6900000171, 6900000168, 6900000166, 6900000165, 6900000162, 6900000161, 6900000158, 6910000541, 6900000156, 6900000155, 6900000154, 6900000153, 6900000152, 6900000150, 6900000149, 6900000145, 6900000144, 6900000143, 6900000142, 6630000003, 6630000002, 6910000438, 6900000054, 6900000049, 6900000047, 6910000327, 6910000308, 6910000306, 6910000303, 6910000301, 6910000299, 6910000298, 6910000294, 6910000293, 6910000292, 6910000291, 6910000287, 6910000283, 6910000282, 6910000281, 6910000279, 6910000276, 1730000020, 6910000274, 6910000272, 6910000271, 6910000343, 2420000001, 6900000002, 2070000002, 6900000045, 6900000044, 6900000041, 6900000040, 6900000035, 6900000034, 6900000029, 6900000028, 6900000025, 6900000021, 6900000003, 6900000020, 6900000019, 6900000014, 6900000013, 6900000012, 6900000011, 6900000010, 6100000001, 2900000002, 6900000172, 6900000199, 1730000007, 6620000076, 6620000062, 6620000056, 6620000055, 6620000045, 6620000022, 1660000013, 1660000012, 1660000017, 1660000016, 6780000011, 2460000014, 1660000006, 2460000012, 2460000011, 1660000005, 2460000009, 2460000008, 2460000007, 2460000006, 2460000005, 2460000002, 6620000063, 6620000079, 6900000200, 6620000082, 6620000152, 6620000151, 6620000145, 2030000007, 2190000001, 1710000005, 2350000002, 2190000003, 2030000004, 6030000003, 2190000002, 2350000001, 6620000101, 6620000098, 6620000097, 6620000096, 6620000095, 6620000093, 6620000092, 6620000091, 6620000085, 2460000001, 3000000211, 6090000064, 3000000184, 6840000007, 6840000006, 6840000005, 6840000004, 1400000005, 1240000005, 1400000004, 1240000003, 6900000250, 6900000244, 6900000242, 6900000240, 6900000238, 6900000223, 6900000221, 6900000220, 6900000215, 6900000214, 6900000210, 6900000205, 6900000202, 6900000268, 6900000269, 6900000275, 3000000137, 1750000008, 3000000173, 3000000172, 1770000018, 1770000016, 10000014, 3000000141, 3000000139, 1290000008, 1610000008, 6900000279, 1610000002, 6570000002, 1930000001, 3000000127, 6040000097, 6040000078, 6040000077, 6040000068, 6900000298, 6240000006, 6910000265, 6910000262, 1010000297, 5000000129, 5000000128, 1160000113, 5000000105, 1160000101, 1160000059, 564, 1160000040, 1160000033, 5000000023, 1160000020, 2440000014, 2760000005, 5000000001, 6760000001, 2110000003, 6070000097, 1010000353, 6070000096, 6070000074, 1010000324, 5000000130, 6010000002, 6010000003, 1690000042, 1160000200, 1690000069, 1690000066, 1210000067, 1210000066, 1210000065, 1160000190, 1160000188, 1690000054, 1690000041, 1210000005, 1690000033, 1690000030, 1160000156, 1690000027, 1210000018, 6650000011, 1210000010, 1210000009, 1210000006, 1010000298, 6070000038, 1160000227, 1270000038, 6180000002, 6180000001, 2340000001, 1010000110, 1010000105, 1010000098, 1010000088, 1010000058, 1010000048, 1010000032, 1010000029, 6130000007, 1810000005, 2450000003, 2450000002, 2130000001, 1010000002, 2130000002, 2450000001, 2720000013, 2720000011, 6180000003, 1010000137, 6500000013, 6550000001, 1750000010, 1270000016, 1270000015, 1270000014, 2390000005, 2390000004, 2390000003, 2390000002, 6550000002, 2230000003, 6500000014, 2230000002, 2230000001, 6710000004, 6710000002, 2070000003, 6710000001, 2070000001, 6340000019, 6500000015, 1690000098, 6620000159, 2540000001, 6910000132, 1840000001, 6910000129, 6910000128, 6910000127, 6910000121, 6910000120, 6910000119, 6910000116, 6910000080, 6910000076, 6910000069, 6910000067, 6910000065, 6910000063, 6910000060, 6910000056, 6910000054, 6910000051, 6910000047, 6910000044, 6910000041, 1840000003, 1840000002, 6910000021, 6910000134, 6910000260, 1570000001, 6910000256, 6910000255, 6910000169, 6910000167, 6910000161, 6910000160, 6910000159, 6910000158, 6910000155, 6910000154, 6910000153, 6910000152, 6910000150, 6910000148, 6910000145, 6910000144, 6910000140, 6910000137, 6910000135, 6910000038, 6910000029, 1160000299, 6700000001, 1160000321, 6910000020, 1690000168, 1690000167, 1690000164, 1160000291, 1690000158, 1690000156, 1160000284, 1740000001, 1690000155, 1160000272, 1690000143, 1690000142, 1690000140, 1580000005, 1690000129, 1580000004, 1580000003, 1690000206, 2010000132, 1690000209, 6910000002, 6910000018, 1900000002, 2390000001, 6910000017, 6910000016, 2110000004, 6910000004, 6910000003, 6910000001, 2430000001, 6860000001, 6590000002, 6590000001, 2430000006, 2430000005, 2430000004, 2430000003, 2430000002, 6700000002]
    to_unique_id = [str(num).zfill(10) for num in unique_id]
    df = get_preprocessing_data3(to_unique_id)
    df = df.groupby('id').apply(prev_30m_generation).reset_index(level=0, drop=True)
    df.dropna(subset=["year","prev_30m_generation"],inplace=True)

    df = df.merge(id_all_data,on=["id"],how="left")
    df["nv2"] = df["generation"] / df["observed_max2"]
    return df,unique_id

### データの分割
#日付に応じてdata_listを作成(LGBM001)
def create_time_series_data(start_date, end_date):

    start_date = datetime.strptime(start_date, "%Y%m%d%H%M")
    end_date = datetime.strptime(end_date, "%Y%m%d%H%M")
    delta = timedelta(minutes=30)

    date_list = []
    date = start_date
    while date <= end_date:
        if date.hour > 6 and date.hour < 18:
            date_list.append(date.strftime("%Y%m%d%H%M"))
        elif date.hour == 18 and date.minute == 0:
            date_list.append(date.strftime("%Y%m%d%H%M"))
        date += delta
    
    return date_list

#↑で6:30も含める(EDA2001)
def create_time_series_data2(start_date, end_date):

    start_date = datetime.strptime(start_date, "%Y%m%d%H%M")
    end_date = datetime.strptime(end_date, "%Y%m%d%H%M")
    delta = timedelta(minutes=30)

    date_list = []
    date = start_date
    while date <= end_date:
        if date.hour > 6 and date.hour < 18:
            date_list.append(date.strftime("%Y%m%d%H%M"))
        elif date.hour == 18 and date.minute == 0:
            date_list.append(date.strftime("%Y%m%d%H%M"))
        elif date.hour == 6 and date.minute == 30:
            date_list.append(date.strftime("%Y%m%d%H%M"))
        date += delta
    
    return date_list


## 誤差計算

#(LGBM002)
def compute_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

#(LGBM002)
def compute_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae





## モデルをトレーニング

#LGBM002
def train_lgbm(X_train, y_train, X_valid, y_valid, lgb_params):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    model = lgb.train(lgb_params, lgb_train , valid_sets=lgb_valid, keep_training_booster=True)
    return model

## 結果の整理
# duration_monthsを計算(result2001)
def get_result1(df):
    df['train_start_date'] = pd.to_datetime(df['train_start_date'], format='%Y%m%d%H%M')
    df['train_end_date'] = pd.to_datetime(df['train_end_date'], format='%Y%m%d%H%M')
    df['duration_months'] = df.apply(lambda row: relativedelta.relativedelta(row['train_end_date'], row['train_start_date']).months, axis=1)

    df['test_start_date'] = pd.to_datetime(df['test_start_date'], format='%Y%m%d%H%M')
    df['test_month'] = df['test_start_date'].dt.month
    return df