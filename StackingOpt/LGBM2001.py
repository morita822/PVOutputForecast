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

    #データセットの読み込み,EDA2001より期間内に実測値、予測値に欠損のないPVをunique_idにしている。
    id_all_data = pd.read_csv("H:\study\output\StackingOpt\EDA006\id_all_data.csv")

    unique_id = [6240000001, 1670000003, 6310000002, 6310000001, 6900000114, 6900000110, 6900000109, 6900000108, 6900000105, 6900000104, 6900000094, 6900000093, 6900000090, 2100000086, 2100000085, 2100000084, 6900000081, 6900000080, 2100000068, 2100000067, 2100000066, 6910000449, 6910000448, 6900000131, 6950000001, 6900000173, 6900000134, 6900000171, 6900000168, 6900000166, 6900000165, 6900000162, 6900000161, 6900000158, 6910000541, 6900000156, 6900000155, 6900000154, 6900000153, 6900000152, 6900000150, 6900000149, 6900000145, 6900000144, 6900000143, 6900000142, 6630000003, 6630000002, 6910000438, 6900000054, 6900000049, 6900000047, 6910000327, 6910000308, 6910000306, 6910000303, 6910000301, 6910000299, 6910000298, 6910000294, 6910000293, 6910000292, 6910000291, 6910000287, 6910000283, 6910000282, 6910000281, 6910000279, 6910000276, 1730000020, 6910000274, 6910000272, 6910000271, 6910000343, 2420000001, 6900000002, 2070000002, 6900000045, 6900000044, 6900000041, 6900000040, 6900000035, 6900000034, 6900000029, 6900000028, 6900000025, 6900000021, 6900000003, 6900000020, 6900000019, 6900000014, 6900000013, 6900000012, 6900000011, 6900000010, 6100000001, 2900000002, 6900000172, 6900000199, 1730000007, 6620000076, 6620000062, 6620000056, 6620000055, 6620000045, 6620000022, 1660000013, 1660000012, 1660000017, 1660000016, 6780000011, 2460000014, 1660000006, 2460000012, 2460000011, 1660000005, 2460000009, 2460000008, 2460000007, 2460000006, 2460000005, 2460000002, 6620000063, 6620000079, 6900000200, 6620000082, 6620000152, 6620000151, 6620000145, 2030000007, 2190000001, 1710000005, 2350000002, 2190000003, 2030000004, 6030000003, 2190000002, 2350000001, 6620000101, 6620000098, 6620000097, 6620000096, 6620000095, 6620000093, 6620000092, 6620000091, 6620000085, 2460000001, 3000000211, 6090000064, 3000000184, 6840000007, 6840000006, 6840000005, 6840000004, 1400000005, 1240000005, 1400000004, 1240000003, 6900000250, 6900000244, 6900000242, 6900000240, 6900000238, 6900000223, 6900000221, 6900000220, 6900000215, 6900000214, 6900000210, 6900000205, 6900000202, 6900000268, 6900000269, 6900000275, 3000000137, 1750000008, 3000000173, 3000000172, 1770000018, 1770000016, 10000014, 3000000141, 3000000139, 1290000008, 1610000008, 6900000279, 1610000002, 6570000002, 1930000001, 3000000127, 6040000097, 6040000078, 6040000077, 6040000068, 6900000298, 6240000006, 6910000265, 6910000262, 1010000297, 5000000129, 5000000128, 1160000113, 5000000105, 1160000101, 1160000059, 564, 1160000040, 1160000033, 5000000023, 1160000020, 2440000014, 2760000005, 5000000001, 6760000001, 2110000003, 6070000097, 1010000353, 6070000096, 6070000074, 1010000324, 5000000130, 6010000002, 6010000003, 1690000042, 1160000200, 1690000069, 1690000066, 1210000067, 1210000066, 1210000065, 1160000190, 1160000188, 1690000054, 1690000041, 1210000005, 1690000033, 1690000030, 1160000156, 1690000027, 1210000018, 6650000011, 1210000010, 1210000009, 1210000006, 1010000298, 6070000038, 1160000227, 1270000038, 6180000002, 6180000001, 2340000001, 1010000110, 1010000105, 1010000098, 1010000088, 1010000058, 1010000048, 1010000032, 1010000029, 6130000007, 1810000005, 2450000003, 2450000002, 2130000001, 1010000002, 2130000002, 2450000001, 2720000013, 2720000011, 6180000003, 1010000137, 6500000013, 6550000001, 1750000010, 1270000016, 1270000015, 1270000014, 2390000005, 2390000004, 2390000003, 2390000002, 6550000002, 2230000003, 6500000014, 2230000002, 2230000001, 6710000004, 6710000002, 2070000003, 6710000001, 2070000001, 6340000019, 6500000015, 1690000098, 6620000159, 2540000001, 6910000132, 1840000001, 6910000129, 6910000128, 6910000127, 6910000121, 6910000120, 6910000119, 6910000116, 6910000080, 6910000076, 6910000069, 6910000067, 6910000065, 6910000063, 6910000060, 6910000056, 6910000054, 6910000051, 6910000047, 6910000044, 6910000041, 1840000003, 1840000002, 6910000021, 6910000134, 6910000260, 1570000001, 6910000256, 6910000255, 6910000169, 6910000167, 6910000161, 6910000160, 6910000159, 6910000158, 6910000155, 6910000154, 6910000153, 6910000152, 6910000150, 6910000148, 6910000145, 6910000144, 6910000140, 6910000137, 6910000135, 6910000038, 6910000029, 1160000299, 6700000001, 1160000321, 6910000020, 1690000168, 1690000167, 1690000164, 1160000291, 1690000158, 1690000156, 1160000284, 1740000001, 1690000155, 1160000272, 1690000143, 1690000142, 1690000140, 1580000005, 1690000129, 1580000004, 1580000003, 1690000206, 2010000132, 1690000209, 6910000002, 6910000018, 1900000002, 2390000001, 6910000017, 6910000016, 2110000004, 6910000004, 6910000003, 6910000001, 2430000001, 6860000001, 6590000002, 6590000001, 2430000006, 2430000005, 2430000004, 2430000003, 2430000002, 6700000002]
    to_unique_id = [str(num).zfill(10) for num in unique_id]
    df = utils.get_preprocessing_data3(to_unique_id)
    df = df.groupby('id').apply(utils.prev_30m_generation).reset_index(level=0, drop=True)
    df.dropna(subset=["year","prev_30m_generation"],inplace=True)

    df = df.merge(id_all_data,on=["id"],how="left")
    df["nv2"] = df["generation"] / df["observed_max2"]

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

        if cfg.use_flo_unique_features:
            X_train_flo_unique = utils.get_unique_pred_interpolated(cfg.flo_unique_dir,train_dates,unique_id)
            X_valid_flo_unique = utils.get_unique_pred_interpolated(cfg.flo_unique_dir,valid_dates,unique_id)
            X_test_flo_unique = utils.get_unique_pred_interpolated(cfg.flo_unique_dir,test_dates,unique_id)
            
            X_train_flo_unique = pd.DataFrame(X_train_flo_unique,columns=[f"{i}_flo_unique" for i in range(X_train_flo_unique.shape[1])])
            X_train_flo_unique["datetime"] = pd.to_datetime(train_dates, format="%Y%m%d%H%M")

            X_valid_flo_unique= pd.DataFrame(X_valid_flo_unique,columns=[f"{i}_flo_unique" for i in range(X_valid_flo_unique.shape[1])])
            X_valid_flo_unique["datetime"] = pd.to_datetime(valid_dates, format="%Y%m%d%H%M")

            X_test_flo_unique= pd.DataFrame(X_test_flo_unique,columns=[f"{i}_flo_unique" for i in range(X_test_flo_unique.shape[1])])
            X_test_flo_unique["datetime"] = pd.to_datetime(test_dates, format="%Y%m%d%H%M")

            X_train = X_train.merge(X_train_flo_unique,on=["datetime"],how="left")
            X_valid = X_valid.merge(X_valid_flo_unique,on=["datetime"],how="left")
            X_test = X_test.merge(X_test_flo_unique,on=["datetime"],how="left")


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

    config_module = importlib.import_module("config")
    cfg_classes = {name: cls for name, cls in inspect.getmembers(config_module, inspect.isclass) if name.startswith("CFG")}

    if args.config in cfg_classes:
        cfg = cfg_classes[args.config]()
    else:
        raise ValueError("無効な設定クラス名が指定されました。")

    main(cfg)