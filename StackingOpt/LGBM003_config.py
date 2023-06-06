'''  実験1 
train_start_date = "201406010000"
train_end_date = "201407010000"
test_start_date = "201407010000"
test_end_date = "201408010000"

CFG1 : use_interpolated_features=True
CFG2 : use_interpolated_features=True, use_pred_features = True
CFG3 : use_interpolated_features=True, use_pred_features = True, use_flo_features = True
CFG4 : use_interpolated_features=True, use_pred_features = True, use_flo_lat_features = True
CFG5 : use_interpolated_features=True, use_pred_features = True, use_flo_lon_features = True
CFG6 : use_interpolated_features=True, use_pred_features = True, use_flo_features = True, use_flo_lat_features = True, use_flo_lon_features = True

cd source/hiroki/study_230408/StackingOpt
python LGBM003.py --config CFG1 && python LGBM003.py --config CFG2 && python LGBM003.py --config CFG3 && python LGBM003.py --config CFG4 && python LGBM003.py --config CFG5 && python LGBM003.py --config CFG6
'''

''' 実験2
python LGBM003.py --config CFG7 && python LGBM003.py --config CFG8 && python LGBM003.py --config CFG9 && python LGBM003.py --config CFG10 && python LGBM003.py --config CFG11 && python LGBM003.py --config CFG12

train_start_date = "201404010000"
train_end_date = "201407010000"
test_start_date = "201407010000"
test_end_date = "201408010000"

CFG7 : use_interpolated_features=True
CFG8 : use_interpolated_features=True, use_pred_features = True
CFG9 : use_interpolated_features=True, use_pred_features = True, use_flo_features = True
CFG10 : use_interpolated_features=True, use_pred_features = True, use_flo_lat_features = True
CFG11 : use_interpolated_features=True, use_pred_features = True, use_flo_lon_features = True
CFG12 : use_interpolated_features=True, use_pred_features = True, use_flo_features = True, use_flo_lat_features = True, use_flo_lon_features = True

'''

#saved_colsをobserved_max2にするのを忘れない。

### 実験1
class CFG1:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG1"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201406010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = False
    use_flo_features = False
    use_flo_lat_features = False
    use_flo_lon_features = False

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG2:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG2"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201406010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = False
    use_flo_lat_features = False
    use_flo_lon_features = False

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG3:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG3"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201406010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = True
    use_flo_lat_features = False
    use_flo_lon_features = False

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG4:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG4"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201406010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = False
    use_flo_lat_features = True
    use_flo_lon_features = False

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG5:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG5"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201406010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = False
    use_flo_lat_features = False
    use_flo_lon_features = True

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG6:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG6"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201406010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = True
    use_flo_lat_features = True
    use_flo_lon_features = True

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

### 実験2 
class CFG7:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG7"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201404010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = False
    use_flo_features = False
    use_flo_lat_features = False
    use_flo_lon_features = False

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG8:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG8"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201404010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = False
    use_flo_lat_features = False
    use_flo_lon_features = False

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG9:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG9"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201404010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = True
    use_flo_lat_features = False
    use_flo_lon_features = False

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG10:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG10"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201404010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = False
    use_flo_lat_features = True
    use_flo_lon_features = False

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG11:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG11"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201404010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = False
    use_flo_lat_features = False
    use_flo_lon_features = True

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG12:
    DEBUG = False
    note_num = "StackingOpt/LGBM003/CFG12"
    n_splits = 5  #データの分割
    seed = 42
    
    #model
    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mae',
                'num_boost_round':100000,
                'early_stopping_rounds':100,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 5,
                'device_type': 'gpu',
                'seed':42
                }
    
    #日付
    train_start_date = "201404010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    interpolated_dir = "H:\study\output\StackingOpt\EDA005"
    pred_dir = "H:\study\output\StackingOpt\EDA006"
    flo_dir = 'H:\study\output\StackingOpt\EDA006'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max']

    use_interpolated_features = True
    use_pred_features = True
    use_flo_features = True
    use_flo_lat_features = True
    use_flo_lon_features = True

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]