""" 実験1、オプティカルフローの予測値有り無しで比較→有りの精度が優れる(result2001)。
cd source/hiroki/study_230408/StackingOpt
python LGBM2001.py --config CFG1 && python LGBM2001.py --config CFG2

6月学習、7月テスト
CFG1 : オプティカルフロー有り
CFG2 : オプティカルフロー無し

"""

""" 実験2, 実験1と同条件で、全てのPVのデータを使用する。
cd source/hiroki/study_230408/StackingOpt
python LGBM2002.py --config CFG3 
"""

""" 実験3, 実験1,2で学習期間3ヵ月
cd source/hiroki/study_230408/StackingOpt
python LGBM2002.py --config CFG4 && python LGBM2002.py --config CFG5 && python LGBM2002.py --config CFG6

4~6月学習,7月テスト
CFG4 : オプ有り 390台
CFG5 : オプ無し 390台
CFG6 : オプ無し,4885台

"""

""" 実験4, 実験1,2で学習期間5ヵ月
cd source/hiroki/study_230408/StackingOpt
python LGBM2002.py --config CFG7 && python LGBM2002.py --config CFG8 && python LGBM2002.py --config CFG9

2~6月学習,7月テスト
CFG7 : オプ有り 390台
CFG8 : オプ無し 390台
CFG9 : オプ無し,4885台

"""

""" 実験5,実験1~4をテストデータを6月にして行う。
cd source/hiroki/study_230408/StackingOpt
sh exam.sh

CFG10~CFG18
"""

""" 実験6,実験1~4をテストデータを5月にして行う。
cd source/hiroki/study_230408/StackingOpt
sh exam.sh

CFG19~CFG27
"""

#
class CFG1:
    DEBUG = False
    #note_num = "StackingOpt/LGBM003/CFG1"
    note_num = "StackingOpt/LGBM2001/CFG1"
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
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG2:
    DEBUG = False
    #note_num = "StackingOpt/LGBM003/CFG2"
    note_num = "StackingOpt/LGBM2001/CFG2"
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
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

# LGBM2002
class CFG3:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG3"
    load_data_kind = "load_data1"
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
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

 # 実験3
class CFG4:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG4"
    load_data_kind = "load_data2"
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
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG5:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG5"
    load_data_kind = "load_data2"
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
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG6:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG6"
    load_data_kind = "load_data1"
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
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

 #実験4
class CFG7:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG7"
    load_data_kind = "load_data2"
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
    train_start_date = "201402010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG8:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG8"
    load_data_kind = "load_data2"
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
    train_start_date = "201402010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG9:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG9"
    load_data_kind = "load_data1"
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
    train_start_date = "201402010000"
    train_end_date = "201407010000"
    test_start_date = "201407010000"
    test_end_date = "201408010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG10:
    DEBUG = False
    #note_num = "StackingOpt/LGBM003/CFG10"
    note_num = "StackingOpt/LGBM2001/CFG10"
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
    train_start_date = "201405010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG11:
    DEBUG = False
    #note_num = "StackingOpt/LGBM003/CFG11"
    note_num = "StackingOpt/LGBM2001/CFG11"
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
    train_start_date = "201405010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

# LGBM2002
class CFG12:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG12"
    load_data_kind = "load_data1"
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
    train_start_date = "201405010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG13:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG13"
    load_data_kind = "load_data2"
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
    train_start_date = "201403010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG14:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG14"
    load_data_kind = "load_data2"
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
    train_start_date = "201403010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG15:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG15"
    load_data_kind = "load_data1"
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
    train_start_date = "201403010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG16:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG16"
    load_data_kind = "load_data2"
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
    train_start_date = "201401010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG17:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG17"
    load_data_kind = "load_data2"
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
    train_start_date = "201401010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG18:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG18"
    load_data_kind = "load_data1"
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
    train_start_date = "201401010000"
    train_end_date = "201406010000"
    test_start_date = "201406010000"
    test_end_date = "201407010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG19:
    DEBUG = False
    #note_num = "StackingOpt/LGBM003/CFG19"
    note_num = "StackingOpt/LGBM2001/CFG19"
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
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG20:
    DEBUG = False
    #note_num = "StackingOpt/LGBM003/CFG20"
    note_num = "StackingOpt/LGBM2001/CFG20"
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
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

# LGBM2002
class CFG21:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG21"
    load_data_kind = "load_data1"
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
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max","generation",target,"pred"]

class CFG22:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG22"
    load_data_kind = "load_data2"
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
    train_start_date = "201402010000"
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG23:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG23"
    load_data_kind = "load_data2"
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
    train_start_date = "201402010000"
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG24:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG24"
    load_data_kind = "load_data1"
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
    train_start_date = "201402010000"
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG25:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG25"
    load_data_kind = "load_data2"
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
    train_start_date = "201312010000"
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = True #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG26:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG26"
    load_data_kind = "load_data2"
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
    train_start_date = "201312010000"
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]

class CFG27:
    DEBUG = False
    note_num = "StackingOpt/LGBM2002/CFG27"
    load_data_kind = "load_data1"
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
    train_start_date = "201312010000"
    train_end_date = "201405010000"
    test_start_date = "201405010000"
    test_end_date = "201406010000"

    #特徴量
    flo_unique_dir = 'H:/study/output/StackingOpt/EDA006/'

    features = ['two_weeks_max', 'id', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos',\
                'prev_30m_generation', 'id_lat', 'id_lng', 'id_lat_mesh', 'id_lng_mesh', 'pvrate', 'observed_max2']


    use_flo_unique_features = False #オプティカルフローのunique_idの予測値

    target = 'nv2'

    #oofで保存するcol
    saved_cols = ["datetime","id","fold","observed_max2","generation",target,"pred"]