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
    if cfg.load_data_kind == "load_data1":
        df,unique_id = utils.load_data1()
    elif cfg.load_data_kind == "load_data2":
        df,unique_id = utils.load_data2()

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

        del X_train,X_valid,X_test,y_train,y_valid,y_test,valid_preds,test_preds

    df.loc[df.datetime.isin(test_dates),"pred"] /= len(train_date_list_split)
    df.loc[df.datetime.isin(train_date_list+test_dates),cfg.saved_cols].to_csv(OUTPUT_DIR+"oof.csv",index=False)

    oof_mse = utils.compute_mse(df.loc[df.datetime.isin(train_date_list),cfg.target] , df.loc[df.datetime.isin(train_date_list),"pred"])
    test_mse = utils.compute_mse(df.loc[df.datetime.isin(test_dates),cfg.target] , df.loc[df.datetime.isin(test_dates),"pred"])

    oof_mae = utils.compute_mae(df.loc[df.datetime.isin(train_date_list),cfg.target] , df.loc[df.datetime.isin(train_date_list),"pred"])
    test_mae = utils.compute_mae(df.loc[df.datetime.isin(test_dates),cfg.target] , df.loc[df.datetime.isin(test_dates),"pred"])

    print('-'*40)
    print(f"note_num: {cfg.note_num}")
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