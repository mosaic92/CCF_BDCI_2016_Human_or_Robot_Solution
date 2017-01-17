import numpy as np
import pandas as pd


# file path
path = '.../AdMaster_competition_dataset/admaster/'

colnames = pd.read_table(path + 'ccf_data_1', nrows=0, sep=',').columns

# media_info_dummy

media_info = pd.read_csv(path + 'ccf_media_info.csv')
media_info['firstType'].replace(['垂直', '门户', '网盟', '其它', '视频', '电商', '搜索', '其他', '移动网盟', '社交', '综合'],
                                ['cz', 'mh', 'wm', 'qt1', 'sp', 'ds', 'ss', 'qt2', 'ydwm', 'sj', 'zh'], inplace=True)
media_dummy = pd.get_dummies(media_info.drop(['secondType', 'tag'], axis=1),
                             columns=['category', 'firstType']).astype(np.int)
media_dummy.to_csv(path + 'media_dummy.csv', index=False)
del media_info

media_dummy = pd.read_csv(path + 'media_dummy.csv')
train_media = pd.read_csv(path + 'ccf_data_3', sep=',',
                          usecols=['global_mediaid'], names=colnames)
train_media = train_media.merge(
    media_dummy, left_on='global_mediaid', right_on='id', how='left')
train_media.drop(['global_mediaid', 'id'], axis=1).to_csv(
    path + 'train3/train3_media.csv', index=False)


# time_transform

train3_time = pd.read_table(path + 'ccf_data_3', sep=',',
                            usecols=['timestamps'], names=colnames)
train3_time['hour'] = pd.to_datetime(train3_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.hour
train3_time['mint'] = pd.to_datetime(train3_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.minute
train3_time['half_hour0'] = (train3_time.mint / 30).astype(np.int)
train3_time['half_hour'] = train3_time.hour * 10 + train3_time.half_hour0
train3_time['minute'] = train3_time.hour * 100 + train3_time.mint
train3_time.to_csv(path + 'train3/train3_' +
                   'time_hour.csv', index=0, columns=['hour'])
train3_time.to_csv(path + 'train3/train3_' +
                   'time_half_hour.csv', index=0, columns=['half_hour'])
train3_time.to_csv(path + 'train3/train3_' +
                   'time_minute.csv', index=0, columns=['minute'])
del train3_time

# feature_cnt

names = [
    'cookie', 'f',
    # 'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid', 'camp',
    # 'creativeid',
    'global_mediaid'
]
for name in names:

    traini = pd.read_table(path + 'ccf_data_3', sep=',',
                           usecols=[name], names=colnames)
    cnt = pd.DataFrame(traini.groupby([name])[name].agg(
        {name + '_cnt': 'count'})).reset_index()
    cnt.loc[:, name + '_index'] = cnt.index
    train_cnt = traini.merge(cnt, on=name, right_index=True).sort_index()
    train_cnt.to_csv(path + 'train3/train3_' + name + '.csv')
    del cnt
    del traini
    del train_cnt
    print name + ': done!'

names = [
    'idfa',
    'mobile_mac', 'mobile_openudid',
    'imei',
    'android_id',
    #     'mobile_os',
    'mobile_type'
]

for name in names:
    df = pd.read_csv(path + 'ccf_data_3', sep=',',
                     usecols=[name], names=colnames)
    df[name + '_cnt'] = df.groupby([name])[name].transform('count').fillna(0)
    df.to_csv(path + 'train3/train3_' + name + '_cnt2.csv')
    del df
    print name + ": done!"


# feature_timestamps_cnt

train3_time = pd.read_table(path + 'ccf_data_3', sep=',',
                            usecols=['timestamps'], names=colnames)
names = [
    'cookie', 'f',
    # 'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid', 'camp',
    # 'creativeid',
    'global_mediaid'
]

for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[3])
    df.loc[:, 'time'] = train3_time.timestamps
    df.loc[:, name + '_time_cnt'] = df.groupby([name + '_index', 'time'])[
        'time'].transform('count')
    df.to_csv(path + 'train3/train3_' + name + '_time_cnt.csv')
    del df
    print name + ': done!'

del train3_time


# feature_cookie_cnt

train3_cookie = pd.read_csv(
    path + 'train3/train3_' + 'cookie' + '.csv', usecols=[3])

names = [
    'f', 'placementid',
    #     'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'camp',
    #     'creativeid',
    'global_mediaid'
]

for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[3])
    df.loc[:, 'cookie'] = train3_cookie.cookie_index
    df.loc[:, name + '_cookie_cnt'] = df.groupby([name + '_index', 'cookie'])[
        'cookie'].transform('count')
    df.to_csv(path + 'train3/train3_' + name + '_cookie_cnt.csv')
    del df
    print name, 'finished'

# feature_f_cnt

train3_f = pd.read_csv(path + 'train3/train3_' + 'f' + '.csv', usecols=[3])
names = [
    'cookie',
    'placementid',
    #     'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'camp',
    #     'creativeid',
    'global_mediaid'
]

for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[3])
    df.loc[:, 'f'] = train3_f.f_index
    df.loc[
        :, name + '_f_cnt'] = df.groupby([name + '_index', 'f'])['f'].transform('count')
    df.to_csv(path + 'train3/train3_' + name + '_f_cnt.csv')
    del df
    print name, 'finished'

del train3_f

# feature_cookie_f_cnt

names = [
    'camp',
    'global_mediaid'
]
train3_cookie = pd.read_csv(
    path + 'train3/train3_' + 'cookie' + '.csv', usecols=[3])
train3_f = pd.read_csv(path + 'train3/train3_' + 'f' + '.csv', usecols=[3])

for name in names:

    df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[3])
    df.loc[:, 'cookie'] = train3_cookie.cookie_index
    df.loc[:, 'f'] = train3_f.f_index

    df.loc[:, name + '_cookie_f_cnt'] = df.groupby(['cookie', 'f', name + '_index'])[
        name + '_index'].transform('count')
    df.to_csv(path + 'train3/train3_' + name + '_cookie_f_cnt.csv')
    print name + ': done!'
    del df

del train3_cookie, train3_f

# feature_hour_cnt

train3_hour = pd.read_csv(path + 'train3/train3_' +
                          'time_hour.csv', usecols=['hour'])

names = [
    'cookie', 'f',
    # 'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid']

for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[3])
    df.loc[:, 'hour'] = train3_hour.hour
    df.loc[:, name + '_hour_cnt'] = df.groupby([name + '_index', 'hour'])[
        'hour'].transform('count')
    df.to_csv(path + 'train3/train3_' + name + '_hour_cnt.csv')
    del df
    print name, 'finished'

del train3_hour

# featur_half_hour_cnt

train3_half_hour = pd.read_csv(
    path + 'train3/train3_' + 'time_half_hour.csv', usecols=['half_hour'])
names = [
    'cookie', 'f',
    #          'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid']

for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[3])
    df.loc[:, 'half_hour'] = train3_half_hour.half_hour
    df.loc[:, name + '_half_hour_cnt'] = df.groupby([name + '_index', 'half_hour'])[
        'half_hour'].transform('count')
    df.to_csv(path + 'train3/train3_' + name + '_half_hour_cnt.csv')
    del df
    print name, 'done!'

del train3_half_hour


# feature_minute_cnt

train3_minute = pd.read_csv(
    path + 'train3/train3_' + 'time_minute.csv', usecols=['minute'])
names = [
    'cookie', 'f',
    #          'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid',
    'global_mediaid'
]

for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name + '.csv', usecols=[3])
    df.loc[:, 'minute'] = train3_minute.minute
    df.loc[:, name + '_minute_cnt'] = df.groupby([name + '_index', 'minute'])[
        'minute'].transform('count')
    df[name + '_minute_cnt_mean'] = df.groupby(
        [name + '_index'])[name + '_minute_cnt'].transform('mean')
    df[name + '_minute_cnt_std'] = df.groupby(
        [name + '_index'])[name + '_minute_cnt'].transform('std')
    df.to_csv(path + 'train3/train3_' + name + '_minute_cnt.csv')
    del df
    print name, 'done!'

del train3_minute


# feature_minute_hour_cnt

names = [
    'cookie',
    'f', 'placementid',
    'global_mediaid',
]
for name in names:
    df = pd.read_csv(path + 'train3/train3_' + name + '_minute_cnt.csv',
                     usecols=[name + '_index', name + '_minute_cnt'])
    df['hour'] = pd.read_csv(path + 'train3/train3_' +
                             'time_hour.csv', usecols=['hour'])
    df[name + '_minute_cnt_hour_mean'] = df.groupby([name + '_index', 'hour'])[
        name + '_minute_cnt'].transform('mean')
    df[name + '_minute_cnt_hour_std'] = df.groupby([name + '_index', 'hour'])[
        name + '_minute_cnt'].transform('std')
    df.to_csv(path + 'train3/train3_' + name + '_hour_stat.csv',
              columns=[name + '_minute_cnt_hour_mean',
                       name + '_minute_cnt_hour_std'],
              index=False)
    del df
    print name, 'done!'
