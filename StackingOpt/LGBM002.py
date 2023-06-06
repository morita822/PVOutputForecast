import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.loss")
import utils

import argparse
import inspect
import importlib

def main(cfg):

    if cfg.DEBUG:
        OUTPUT_DIR = f'H:/study/output/DEBUG/{cfg.note_num}/'
    else:
        OUTPUT_DIR = f'H:/study/output/{cfg.note_num}/'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    utils.set_seeds()
    # 時系列の分割設定
    train_date_list = utils.create_time_series_data(cfg.train_start_date,cfg.train_end_date)
    train_date_list_split = np.array_split(train_date_list, cfg.n_splits)

    test_dates = utils.create_time_series_data(cfg.test_start_date,cfg.test_end_date)

    #データセットの読み込み
    unique_id = ['10000095', '10000269', '1020000002', '1110000001', '1110000010', '1110000011', '1110000012', '1110000013', '1110000014', '1110000015', '1160000025', '1160000090', '1160000091', '1160000182', '1160000185', '1160000253', '1160000387', '1160000402', '1160000419', '1160000420', '1160000423', '1270000026', '1280000048', '1550000001', '1650000004', '1680000001', '1680000002', '1680000003', '1680000004', '1680000010', '1680000017', '1680000021', '1680000033', '1680000047', '1680000054', '1680000057', '1680000063', '1680000067', '1680000080', '1680000081', '1680000097', '1680000107', '1680000108', '1680000112', '1680000151', '1680000152', '1680000213', '1680000216', '1680000217', '1680000218', '1680000223', '1680000228', '1680000285', '1680000287', '1680000327', '1680000364', '2220000001', '2220000002', '2220000003', '2730000001', '2910000002', '3000000007', '3000000012', '3000000042', '5000000044', '5000000045', '6000000016', '6000000017', '6060000016', '6060000017', '6060000018', '6170000016', '6170000123', '6170000124', '6170000125', '6620000065', '6620000066', '6620000088', '6620000089', '6620000111', '6620000117', '6620000118', '6620000121', '6620000122', '6620000123', '6620000124', '6620000131', '6620000132', '6910000180', '6910000198', '6910000200', '6910000206', '6910000216', '6910000217', '6910000239', '6910000240', '6910000249', '6910000250', '6910000421', '6910000424', '6910000425', '6910000469', '6910000470']
    unique_id = [int(i) for i in unique_id]
    to_unique_id = [str(num).zfill(10) for num in unique_id]
    df = utils.get_preprocessing_data(to_unique_id)
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset=["id","datetime"],inplace=True)
    df = df.groupby('id').apply(utils.prev_30m_generation).reset_index(level=0, drop=True)
    id_all_data = pd.read_csv("H:\study\preprocessing_data\id_all_data.csv",encoding='shift_jis')
    df = df.merge(id_all_data,on=["id"],how="left")
    df.dropna(subset=["year"],inplace=True) #utils.prev_30m_generationで30分間隔のデータセットになっているため欠損が出ている。
    df["nv2"] = df["generation"] / df["observed_max"]

    #oof作成用
    df["pred"] = 0
    df.loc[df.datetime.isin(test_dates),"fold"] = "test"
    
    for fold in range(len(train_date_list_split)):
        print(f"\nFold {fold + 1}")
        train_dates = np.concatenate(train_date_list_split[:fold] + train_date_list_split[fold+1:])
        valid_dates = train_date_list_split[fold]

        X_train, y_train = df.loc[df.datetime.isin(train_dates),cfg.features+["datetime"]],df.loc[df.datetime.isin(train_dates),cfg.target]
        X_valid, y_valid = df.loc[df.datetime.isin(valid_dates),cfg.features+["datetime"]],df.loc[df.datetime.isin(valid_dates),cfg.target]
        X_test, y_test = df.loc[df.datetime.isin(test_dates),cfg.features+["datetime"]],df.loc[df.datetime.isin(test_dates),cfg.target]
        df.loc[df.datetime.isin(valid_dates),"fold"] = fold

        if cfg.use_interpolated_features:
            X_train_interpolated, y_train_interpolated = utils.get_interpolated_mesh_data(cfg.interpolated_dir,train_dates)
            X_valid_interpolated, y_valid_interpolated = utils.get_interpolated_mesh_data(cfg.interpolated_dir,valid_dates)
            X_test_interpolated, y_test_interpolated = utils.get_interpolated_mesh_data(cfg.interpolated_dir,test_dates)
            
            X_train_interpolated = pd.DataFrame(X_train_interpolated)
            X_train_interpolated["datetime"] = pd.to_datetime(train_dates, format="%Y%m%d%H%M")

            X_valid_interpolated = pd.DataFrame(X_valid_interpolated)
            X_valid_interpolated["datetime"] = pd.to_datetime(valid_dates, format="%Y%m%d%H%M")

            X_test_interpolated = pd.DataFrame(X_test_interpolated)
            X_test_interpolated["datetime"] = pd.to_datetime(test_dates, format="%Y%m%d%H%M")

            X_train = X_train.merge(X_train_interpolated,on=["datetime"],how="left")
            X_valid = X_valid.merge(X_valid_interpolated,on=["datetime"],how="left")
            X_test = X_test.merge(X_test_interpolated,on=["datetime"],how="left")

        if cfg.use_pred_features:
            X_train_interpolated_pred = utils.get_pred_interpolated_mesh_data(cfg.pred_dir,train_dates)
            X_valid_interpolated_pred = utils.get_pred_interpolated_mesh_data(cfg.pred_dir,valid_dates)
            X_test_interpolated_pred = utils.get_pred_interpolated_mesh_data(cfg.pred_dir,test_dates)
            
            X_train_interpolated_pred  = pd.DataFrame(X_train_interpolated_pred )
            X_train_interpolated_pred ["datetime"] = pd.to_datetime(train_dates, format="%Y%m%d%H%M")

            X_valid_interpolated_pred = pd.DataFrame(X_valid_interpolated_pred)
            X_valid_interpolated_pred["datetime"] = pd.to_datetime(valid_dates, format="%Y%m%d%H%M")

            X_test_interpolated_pred = pd.DataFrame(X_test_interpolated_pred)
            X_test_interpolated_pred["datetime"] = pd.to_datetime(test_dates, format="%Y%m%d%H%M")

            X_train = X_train.merge(X_train_interpolated_pred,on=["datetime"],how="left")
            X_valid = X_valid.merge(X_valid_interpolated_pred,on=["datetime"],how="left")
            X_test = X_test.merge(X_test_interpolated_pred,on=["datetime"],how="left")

        X_train.drop("datetime",axis=1,inplace=True)
        X_valid.drop("datetime",axis=1,inplace=True)
        X_test.drop("datetime",axis=1,inplace=True)

        # Train LightGBM model
        model = utils.train_lgbm(X_train, y_train, X_valid, y_valid, cfg.lgb_params)
        save_path = OUTPUT_DIR + f"/lgbm_fold{fold}.txt"
        model.save_model(save_path)

        # Evaluate model
        valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
        mse = utils.compute_mse(y_valid, valid_preds)
        mae = utils.compute_mae(y_valid, valid_preds)
        print(f"Fold {fold + 1} MSE: {mse}, MAE: {mae}")

        # Make predictions for the test set
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)

        df.loc[df.datetime.isin(valid_dates),"pred"] = valid_preds
        df.loc[df.datetime.isin(test_dates),"pred"] += test_preds


    df.loc[df.datetime.isin(test_dates),"pred"] /= len(train_date_list_split)
    df.loc[df.datetime.isin(train_date_list+test_dates),cfg.saved_cols].to_csv(OUTPUT_DIR+"oof.csv",index=False)

    oof_mse = utils.compute_mse(df.loc[df.datetime.isin(train_date_list),cfg.target] , df.loc[df.datetime.isin(train_date_list),"pred"])
    test_mse = utils.compute_mse(df.loc[df.datetime.isin(test_dates),cfg.target] , df.loc[df.datetime.isin(test_dates),"pred"])

    oof_mae = utils.compute_mae(df.loc[df.datetime.isin(train_date_list),cfg.target] , df.loc[df.datetime.isin(train_date_list),"pred"])
    test_mae = utils.compute_mae(df.loc[df.datetime.isin(test_dates),cfg.target] , df.loc[df.datetime.isin(test_dates),"pred"])

    print('-'*40)
    print(f"Overall Out-of-Fold RMSE: {np.sqrt(oof_mse):.4f}")
    print(f"Overall Out-of-Fold MAE: {oof_mae:.4f}")
    print()
    print(f"Overall Test RMSE: {np.sqrt(test_mse):.4f}")
    print(f"Overall Test MAE: {test_mae:.4f}")
    print('-'*40)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="CFG1", help="設定クラスを選択（CFG1 または CFG2 など）")
    args = parser.parse_args()

    config_module = importlib.import_module("LGBM002_config")
    cfg_classes = {name: cls for name, cls in inspect.getmembers(config_module, inspect.isclass) if name.startswith("CFG")}

    if args.config in cfg_classes:
        cfg = cfg_classes[args.config]()
    else:
        raise ValueError("無効な設定クラス名が指定されました。")

    main(cfg)