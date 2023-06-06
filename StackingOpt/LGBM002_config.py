## LGBM002

class CFG1:
    DEBUG = False
    note_num = "StackingOpt/LGBM002"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':10,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    # train_start_date = "201308150000"
    # train_end_date = "201407010000"
    # test_start_date = "201407010000"
    # test_end_date = "201408010000"

    train_start_date = "201308150000"
    train_end_date = "20130817000"
    test_start_date = "201308170000"
    test_end_date = "20130818000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]