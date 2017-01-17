import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
import lightgbm as lgb


# file path
path = '.../AdMaster_competition_dataset/admaster/'


# colnames
colnames = pd.read_table(path + 'ccf_data_1', nrows=0, sep=',').columns
colnames2 = pd.read_table(path + 'test_data_1_1118', nrows=0, sep=',').columns

################################################## train3 ################

train = pd.read_csv(path + 'ccf_data_3', sep=',',
                    usecols=['label'],
                    dtype={'label': np.int},
                    names=colnames)

train_media = pd.read_csv(path + 'train3/train3_media.csv',
                          usecols=['firstType_cz', 'firstType_sp'])

train = pd.concat([train, train_media], axis=1)

del train_media

filenames = [
    'cookie', 'f', 'placementid', 'camp', 'global_mediaid',
    'cookie_time_cnt', 'f_time_cnt', 'camp_time_cnt',
    'f_cookie_cnt', 'placementid_cookie_cnt', 'camp_cookie_cnt',
    'cookie_hour_cnt', 'f_hour_cnt', 'placementid_hour_cnt',
    'cookie_half_hour_cnt', 'f_half_hour_cnt', 'placementid_half_hour_cnt',
    'placementid_f_cnt', 'camp_f_cnt', 'global_mediaid_f_cnt'
]

for name in filenames:
    if 'cnt' in name:
        df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[3])
        train[name] = df
        print name + ': done!'
        del df
    else:
        df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[2])
        train[name + 'cnt'] = df
        print name + ': done!'
        del df

names = ['camp', 'global_mediaid']

for name in names:

    df = pd.read_csv(path + 'train3/train3_' + name + '_cookie_f_cnt.csv',
                     usecols=[name + '_cookie_f_cnt'])
    train[name + '_cookie_f_cnt'] = df
    print name + ': done!'
    del df

names = ['f']
for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name +
                     '_minute_cnt.csv', index_col=0)
    train[name + '_minute_cnt_mean'] = df[name + '_minute_cnt_mean']
    train[name + '_minute_cnt_std'] = df[name + '_minute_cnt_std']
    del df
    print name + ': done!'

names = ['f']
for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name + '_hour_stat.csv')
    train[name + '_minute_cnt_hour_mean'] = df[name + '_minute_cnt_hour_mean']
    train[name + '_minute_cnt_hour_std'] = df[name + '_minute_cnt_hour_std']
    del df
    print name + ': done!'


names = ['mobile_mac', 'mobile_openudid', 'imei']

for name in names:
    train[name + '_cnt'] = pd.read_csv(path + 'train3/train3_' + name + '_cnt2.csv',
                                       usecols=[name + '_cnt'])
    print name + ": done!"


train.to_csv(path + 'train3/train3_data_final.csv', index=False)


############################################### training #################

train = pd.read_csv(path + 'train3/train3_data_final.csv')

feat0 = [4, 22, 23, 19, 7, 6, 5, 20, 15, 3, 1,
         11, 25, 13, 9, 27, 2, 10, 14, 16, 21, 24, 26]
feat1 = [4, 23, 19, 8, 7, 6, 5, 17, 20, 3, 1,
         18, 11, 13, 12, 9, 2, 10, 28, 14, 21, 24, 26]
feat2 = [4, 22, 8, 19, 7, 17, 5, 20, 15, 18,
         1, 11, 25, 13, 9, 27, 28, 10, 14, 16, 24]
# feat3 = [4,8,23,19,7,17,6,5,20,18,3,1,11,12,13,9,28,2,10,14,21,24,26]
# feat4 = [8,22,23,19,17,6,5,18,15,3,1,12,25,13,28,27,2,10,16,21,24,26]

i = 0
for feat in [feat0, feat1, feat2]:

    i = i + 1
    x_train = train.iloc[:, feat]
    y_train = train.label
    lgb_train = (x_train, y_train)
    # lgb_eval = (x_train, y_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'seed': 9527
    }

    def fscore(preds, train_data):
        labels = train_data.get_label()
        return 'F1-score', metrics.f1_score(labels, (preds > 0.5)), True

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    #                 valid_datas=lgb_eval,
                    feval=fscore,
                    #                 early_stopping_rounds=10
                    )
    gbm.save_model(path + 'model_final' + str(i) + '.txt')


##################################################### test1 ##############

test = pd.read_csv(path + 'test/test1' + '_media.csv',
                   usecols=['firstType_cz', 'firstType_sp'])

filenames = [
    'cookie', 'f', 'placementid', 'camp', 'global_mediaid',
    'cookie_time_cnt', 'f_time_cnt', 'camp_time_cnt',
    'f_cookie_cnt', 'placementid_cookie_cnt', 'camp_cookie_cnt',
    'cookie_hour_cnt', 'f_hour_cnt', 'placementid_hour_cnt',
    #   'cookie_half_hour_cnt', 'f_half_hour_cnt', 'placementid_half_hour_cnt',
    'placementid_f_cnt', 'camp_f_cnt', 'global_mediaid_f_cnt'
]

for name in filenames:
    if 'cnt' in name:
        df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[3])
        test[name] = df
        del df
        print name + ': done!'
    else:
        df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[2])
        test[name + 'cnt'] = df
        del df
        print name + ': done!'

names = ['camp', 'global_mediaid']

for name in names:

    df = pd.read_csv(path + 'test/test1_' + name + '_cookie_f_cnt.csv',
                     usecols=[name + '_cookie_f_cnt'])
    test[name + '_cookie_f_cnt'] = df
    print name + ': done!'
    del df

names = [
    'f',
]
for name in names:
    df = pd.read_csv(path + 'test/test1_' + name +
                     '_minute_cnt.csv', index_col=0)
    test[name + '_minute_cnt_mean'] = df[name + '_minute_cnt_mean']
    test[name + '_minute_cnt_std'] = df[name + '_minute_cnt_std']
    del df
    print name + ': done!'


names = [
    'f',
]
for name in names:
    df = pd.read_csv(path + 'test/test1_' + name + '_hour_stat.csv')
    test[name + '_minute_cnt_hour_mean'] = df[name + '_minute_cnt_hour_mean']
    test[name + '_minute_cnt_hour_std'] = df[name + '_minute_cnt_hour_std']
    del df
    print name + ': done!'

names = [
    'mobile_mac', 'mobile_openudid', 'imei',
]

for name in names:
    test[name + '_cnt'] = pd.read_csv(path + 'test/test1_' + name + '_cnt2.csv',
                                      usecols=[name + '_cnt'])
    print name + ": done!"

test.to_csv(path + 'test/test1_data_final.csv', index=False)

################################################### test2 ################

test2 = pd.read_csv(path + 'test/test2' + '_media.csv',
                    usecols=['firstType_cz', 'firstType_sp'])

filenames = [
    'cookie', 'f', 'placementid', 'camp', 'global_mediaid',
    'cookie_time_cnt', 'f_time_cnt', 'camp_time_cnt',
    'f_cookie_cnt', 'placementid_cookie_cnt', 'camp_cookie_cnt',
    'cookie_hour_cnt', 'f_hour_cnt', 'placementid_hour_cnt',
    # 'cookie_half_hour_cnt', 'f_half_hour_cnt', 'placementid_half_hour_cnt',
    'placementid_f_cnt', 'camp_f_cnt', 'global_mediaid_f_cnt'
]

for name in filenames:
    if 'cnt' in name:
        df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[3])
        test2[name] = df
        del df
    else:
        df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[2])
        test2[name + 'cnt'] = df
        del df

names = ['camp', 'global_mediaid']

for name in names:

    df = pd.read_csv(path + 'test/test2_' + name + '_cookie_f_cnt.csv',
                     usecols=[name + '_cookie_f_cnt'])
    test2[name + '_cookie_f_cnt'] = df
    print name + ': done!'
    del df

names = [
    'f',
]
for name in names:
    df = pd.read_csv(path + 'test/test2_' + name +
                     '_minute_cnt.csv', index_col=0)
    test2[name + '_minute_cnt_mean'] = df[name + '_minute_cnt_mean']
    test2[name + '_minute_cnt_std'] = df[name + '_minute_cnt_std']
    del df
    print name + ': done!'


names = [
    'f',
]
for name in names:
    df = pd.read_csv(path + 'test/test2_' + name + '_hour_stat.csv')
    test2[name + '_minute_cnt_hour_mean'] = df[name + '_minute_cnt_hour_mean']
    test2[name + '_minute_cnt_hour_std'] = df[name + '_minute_cnt_hour_std']
    del df
    print name + ': done!'


names = [
    'mobile_mac', 'mobile_openudid', 'imei',
]

for name in names:
    test2[name + '_cnt'] = pd.read_csv(path + 'test/test2_' + name + '_cnt2.csv',
                                       usecols=[name + '_cnt'])
    print name + ": done!"

test.to_csv(path + 'test/test1_data_final.csv', index=False)

########################################### prediction ###################

for i in range(3):

    gbm = lgb.Booster(model_file=path + 'model_final' + str(i) + '.txt')

    j = 0

    for testi in pd.read_csv(path + 'test/test1_data_final.csv', chunksize=10000000):
        j += 1
        testi = testi.iloc[:, list(np.array(feat2) - 1)]
        pred = gbm.predict(testi, num_iteration=gbm.best_iteration)
        pred = pd.DataFrame({'pred': pred})
        pred.to_csv(path + 'test/test1_pred_' + str(i) +
                    '.csv', mode='a', index=False, header=False)
        print 'part', j, ": done!"
        del pred

    j = 0

    for testi in pd.read_csv(path + 'test/test2_data_final.csv', chunksize=10000000):
        j += 1
        testi = testi.iloc[:, list(np.array(feat2) - 1)]
        pred = gbm.predict(testi, num_iteration=gbm.best_iteration)
        pred = pd.DataFrame({'pred': pred})
        pred.to_csv(path + 'test/test2_pred_' + str(i) +
                    '.csv', mode='a', index=False, header=False)
        print 'part', j, ": done!"
        del pred

pred10 = pd.read_csv(path + 'test/test1_pred_0.csv', header=None)
pred11 = pd.read_csv(path + 'test/test1_pred_1.csv', header=None)
pred12 = pd.read_csv(path + 'test/test1_pred_2.csv', header=None)
pred1 = (pred10.iloc[:, 0] + pred11.iloc[:, 0] + pred12.iloc[:, 0]) / 3
pred1 = pd.DataFrame({'pred': pred1})

pred20 = pd.read_csv(path + 'test/test2_pred_0.csv', header=None)
pred21 = pd.read_csv(path + 'test/test2_pred_1.csv', header=None)
pred22 = pd.read_csv(path + 'test/test2_pred_2.csv', header=None)
pred2 = (pred20.iloc[:, 0] + pred21.iloc[:, 0] + pred22.iloc[:, 0]) / 3
pred2 = pd.DataFrame({'pred': pred2})

pred = pd.concat([pred1, pred2], axis=0).reset_index().drop(
    ['index'], axis=1).reset_index()

sub = pred.loc[pred.pred > 0.5, 'index']

sub.to_csv(path + 'sub_final.csv', index=False, header=None)
