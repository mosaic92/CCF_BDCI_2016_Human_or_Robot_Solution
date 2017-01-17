import numpy as np
import pandas as pd

# file path
path = '.../AdMaster_competition_dataset/admaster/'

# colnames
colnames = pd.read_table(path + 'ccf_data_1', nrows=0, sep=',').columns
colnames2 = pd.read_table(path + 'test_data_1_1118', nrows=0, sep=',').columns

# split and combine test set
names = [
    'cookie', 'f', 'timestamps',
    'play',
    'channel',
    'idfa', 'mobile_mac', 'mobile_openudid', 'imei',
    'android_id', 'mobile_os',
    'mobile_type', 'mobile_app_key',
    'mobile_app_name', 'creativeid', 'placementid', 'global_mediaid', 'camp'
]
for name in names:
    test1 = pd.read_table(path + 'test_data_1_1118', sep=',', usecols=[name])
    test2 = pd.read_table(path + 'test_data_2_1118',
                          sep=',', names=colnames2, usecols=[name])
    testi = pd.concat([test1, test2], axis=0)
    testi.to_csv(path + 'test_' + name + '.csv', index=False)

    del test1, test2, testi
    print name, ": finished"


############################################ test1 #######################

# time_transform

test1_time = pd.read_table(path + 'test_timestamps.csv', sep=',',
                           usecols=['timestamps'], nrows=64814822)
test1_time['hour'] = pd.to_datetime(test1_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.hour
test1_time['mint'] = pd.to_datetime(test1_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.minute
test1_time['half_hour0'] = (test1_time.mint / 30).astype(np.int)
test1_time['half_hour'] = test1_time.hour * 10 + test1_time.half_hour0
test1_time['minute'] = test1_time.hour * 100 + test1_time.mint
test1_time.to_csv(path + 'test/test1_' + 'time_hour.csv',
                  index=0, columns=['hour'])
test1_time.to_csv(path + 'test/test1_' + 'time_half_hour.csv',
                  index=0, columns=['half_hour'])
test1_time.to_csv(path + 'test/test1_' + 'time_minute.csv', columns=['minute'])
del test1_time


# feature_cnt

names = [
    'cookie', 'f',
    # 'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid', 'camp',
    # 'creativeid',
    'global_mediaid'
]
for name in names:

    testi = pd.read_table(path + 'test_' + name + '.csv', sep=',',
                          usecols=[name], nrows=64814822)
    cnt = pd.DataFrame(testi.groupby([name])[name].agg(
        {name + '_cnt': 'count'})).reset_index()
    cnt.loc[:, name + '_index'] = cnt.index
    test_cnt = testi.merge(cnt, on=name, right_index=True).sort_index()
    test_cnt.to_csv(path + 'test/test1_' + name + '.csv')
    del cnt
    del testi
    print name + ' finished'

names = [
    'idfa',
    'mobile_mac', 'mobile_openudid',
    'imei',
    'android_id',
    #     'mobile_os',
    'mobile_type'
]

for name in names:
    df = pd.read_table(path + 'test_' + name + '.csv', sep=',',
                       usecols=[name], nrows=64814822)
    df[name + '_cnt'] = df.groupby([name])[name].transform('count').fillna(0)
    df.to_csv(path + 'test3/test1_' + name + '_cnt2.csv')
    del df
    print name + ": done!"

# feature_timestamps_cnt

test1_time = pd.read_table(path + 'test_timestamps.csv', sep=',',
                           usecols=['timestamps'], nrows=64814822)
names = [
    'cookie', 'f', 'placementid',
    'camp', ]

for name in names:
    df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[3])
    df.loc[:, 'date'] = test1_time.timestamps
    df.loc[:, name + '_date_cnt'] = df.groupby([name + '_index', 'date'])[
        'date'].transform('count')
    df.to_csv(path + 'test/test1_' + name + '_time_cnt.csv')
    del df
    print name, 'finished'

# feature_cookie_cnt

test1_cookie = pd.read_csv(path + 'test/test1_' +
                           'cookie' + '.csv', usecols=[3])

names = [
    'f', 'placementid',
    'camp']

for name in names:
    df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[3])
    df.loc[:, 'cookie'] = test1_cookie.cookie_index
    df.loc[:, name + '_cookie_cnt'] = df.groupby([name + '_index', 'cookie'])[
        'cookie'].transform('count')
    df.to_csv(path + 'test/test1_' + name + '_cookie_cnt.csv')
    del df
    print name, 'finished'


# feature_f_cnt

test1_f = pd.read_csv(path + 'test/test1_' + 'f' + '.csv', usecols=[3])

names = [
    #     'cookie', 'placementid','camp',
    'global_mediaid']

for name in names:
    df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[3])
    df.loc[:, 'f'] = test1_f.f_index
    df.loc[
        :, name + '_f_cnt'] = df.groupby([name + '_index', 'f'])['f'].transform('count')
    df.to_csv(path + 'test/test1_' + name + '_f_cnt.csv')
    print name, 'finished'

# feature_cookie_f_cnt

names = [
    'camp',
    'global_mediaid']
test1_cookie = pd.read_csv(path + 'test/test1_' +
                           'cookie' + '.csv', usecols=[3])
test1_f = pd.read_csv(path + 'test/test1_' + 'f' + '.csv', usecols=[3])

for name in names:

    df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[3])
    df.loc[:, 'cookie'] = test1_cookie.cookie_index
    df.loc[:, 'f'] = test1_f.f_index

    df.loc[:, name + '_cookie_f_cnt'] = df.groupby(['cookie', 'f', name + '_index'])[
        name + '_index'].transform('count')
    df.to_csv(path + 'test/test1_' + name + '_cookie_f_cnt.csv')
    print name + ': done!'
    del df

del test1_cookie, test1_f

# feature_hour_cnt

test1_time = pd.read_table(path + 'test_timestamps.csv', sep=',',
                           usecols=['timestamps'], nrows=64814822)
test1_time['hour'] = pd.to_datetime(test1_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.hour
names = ['cookie', 'f', 'placementid']

for name in names:
    df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[3])
    df.loc[:, 'hour'] = test1_time.hour
    df.loc[:, name + '_hour_cnt'] = df.groupby([name + '_index', 'hour'])[
        'hour'].transform('count')
    df.to_csv(path + 'test/test1_' + name + '_hour_cnt.csv')
    del df
    print name, 'finished'


# featur_half_hour_cnt

test1_time = pd.read_table(path + 'test_timestamps.csv', sep=',',
                           usecols=['timestamps'], nrows=64814822)
test1_time['hour'] = pd.to_datetime(test1_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.hour
test1_time['mint'] = pd.to_datetime(test1_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.minute
test1_time['mint'] = (test1_time.mint / 30).astype(np.int)
test1_time['half_hour'] = test1_time.hour * 10 + test1_time.mint
names = ['cookie', 'f', 'placementid']

for name in names:
    df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[3])
    df.loc[:, 'half_hour'] = test1_time.half_hour
    df.loc[:, name + '_half_hour_cnt'] = df.groupby([name + '_index', 'half_hour'])[
        'half_hour'].transform('count')
    df.to_csv(path + 'test/test1_' + name + '_half_hour_cnt.csv')
    del df
    print name, 'finished'

# feature_minute_cnt

test1_minute = pd.read_csv(path + 'test/test1_' +
                           'time_minute.csv', usecols=['minute'])
names = [
    'cookie', 'f',
    #          'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid', ]

for name in names:
    df = pd.read_csv(path + 'test/test1_' + name + '.csv', usecols=[3])
    df.loc[:, 'minute'] = test1_minute.minute
    df.loc[:, name + '_minute_cnt'] = df.groupby([name + '_index', 'minute'])[
        'minute'].transform('count')
    df[name + '_minute_cnt_mean'] = df.groupby(
        [name + '_index'])[name + '_minute_cnt'].transform('mean')
    df[name + '_minute_cnt_std'] = df.groupby(
        [name + '_index'])[name + '_minute_cnt'].transform('std')
    df.to_csv(path + 'test/test1_' + name + '_minute_cnt.csv')
    del df
    print name, 'done!'

del test1_minute


# feature_minute_hour_cnt

names = [
    'cookie',
    'f',
    'placementid',
    'global_mediaid',
]
for name in names:
    df = pd.read_csv(path + 'test/test1_' + name + '_minute_cnt.csv',
                     usecols=[name + '_index', name + '_minute_cnt'])
    df['hour'] = pd.read_csv(path + 'test/test1_' +
                             'time_hour.csv', usecols=['hour'])
    df[name + '_minute_cnt_hour_mean'] = df.groupby([name + '_index', 'hour'])[
        name + '_minute_cnt'].transform('mean')
    df[name + '_minute_cnt_hour_std'] = df.groupby([name + '_index', 'hour'])[
        name + '_minute_cnt'].transform('std')
    df.to_csv(path + 'test/test1_' + name + '_hour_stat.csv',
              columns=[name + '_minute_cnt_hour_mean',
                       name + '_minute_cnt_hour_std'],
              index=False)
    del df
    print name, 'done!'

# media_info

media_dummy = pd.read_csv(path + 'media_dummy.csv')
test_media = pd.read_table(path + 'test_global_mediaid.csv', sep=',',
                           usecols=['global_mediaid'], nrows=64814822)
test_media = test_media.merge(
    media_dummy, left_on='global_mediaid', right_on='id', how='left')
test_media.drop(['global_mediaid', 'id'], axis=1).to_csv(
    path + 'test/test1_media.csv', index=False)


############################################ test2 #######################

# time_transform

test2_time = pd.read_table(path + 'test_timestamps.csv', sep=',',
                           skiprows=64814823, names=['timestamps'])
test2_time['hour'] = pd.to_datetime(test2_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.hour
test2_time['mint'] = pd.to_datetime(test2_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.minute
test2_time['half_hour0'] = (test2_time.mint / 30).astype(np.int)
test2_time['half_hour'] = test2_time.hour * 10 + test2_time.half_hour0
test2_time['minute'] = test2_time.hour * 100 + test2_time.mint
test2_time.to_csv(path + 'test/test2_' + 'time_hour.csv',
                  index=0, columns=['hour'])
test2_time.to_csv(path + 'test/test2_' + 'time_half_hour.csv',
                  index=0, columns=['half_hour'])
test2_time.to_csv(path + 'test/test2_' + 'time_minute.csv', columns=['minute'])

# feature_cnt

names = [
    'cookie', 'f',
    # 'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid',
    'camp', 'global_mediaid'
]
for name in names:

    testi = pd.read_table(path + 'test_' + name + '.csv', sep=',',
                          names=[name], skiprows=64814823)
    cnt = pd.DataFrame(testi.groupby([name])[name].agg(
        {name + '_cnt': 'count'})).reset_index()
    cnt.loc[:, name + '_index'] = cnt.index
    test_cnt = testi.merge(cnt, on=name, right_index=True).sort_index()
    test_cnt.to_csv(path + 'test/test2_' + name + '.csv')
    del cnt
    del testi
    print name + ' finished'


# feature_timestamps_cnt

test2_time = pd.read_table(path + 'test_' + 'timestamps' + '.csv', sep=',',
                           names=['timestamps'], skiprows=64814823)

names = [
    'cookie', 'f', 'placementid',
    'camp', 'global_mediaid']

for name in names:
    df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[3])
    df.loc[:, 'date'] = test2_time.timestamps
    df.loc[:, name + '_date_cnt'] = df.groupby([name + '_index', 'date'])[
        'date'].transform('count')
    df.to_csv(path + 'test/test2_' + name + '_time_cnt.csv')
    del df
    print name, 'finished'


# feature_cookie_cnt

test2_cookie = pd.read_csv(path + 'test/test2_' +
                           'cookie' + '.csv', usecols=[3])

names = [
    'f', 'placementid',
    'camp']

for name in names:
    df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[3])
    df.loc[:, 'cookie'] = test2_cookie.cookie_index
    df.loc[:, name + '_cookie_cnt'] = df.groupby([name + '_index', 'cookie'])[
        'cookie'].transform('count')
    df.to_csv(path + 'test/test2_' + name + '_cookie_cnt.csv')
    print name, 'finished'

# feature_f_cnt

test2_f = pd.read_csv(path + 'test/test2_' + 'f' + '.csv', usecols=[3])

names = [
    'placementid', 'camp',
    'global_mediaid']

for name in names:
    df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[3])
    df.loc[:, 'f'] = test2_f.f_index
    df.loc[
        :, name + '_f_cnt'] = df.groupby([name + '_index', 'f'])['f'].transform('count')
    df.to_csv(path + 'test/test2_' + name + '_f_cnt.csv')
    print name, 'finished'

# feature_f_cookie_cnt

names = [
    'camp',
    'global_mediaid']
test2_cookie = pd.read_csv(path + 'test/test2_' +
                           'cookie' + '.csv', usecols=[3])
test2_f = pd.read_csv(path + 'test/test2_' + 'f' + '.csv', usecols=[3])

for name in names:

    df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[3])
    df.loc[:, 'cookie'] = test2_cookie.cookie_index
    df.loc[:, 'f'] = test2_f.f_index

    df.loc[:, name + '_cookie_f_cnt'] = df.groupby(['cookie', 'f', name + '_index'])[
        name + '_index'].transform('count')
    df.to_csv(path + 'test/test2_' + name + '_cookie_f_cnt.csv')
    print name + ': done!'
    del df

del test1_cookie, test1_f

# feature_hour_cnt

test2_time = pd.read_table(path + 'test_' + 'timestamps' + '.csv', sep=',',
                           names=['timestamps'], skiprows=64814823)
test2_time['hour'] = pd.to_datetime(test2_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.hour
names = ['cookie', 'f', 'placementid']

for name in names:
    df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[3])
    df.loc[:, 'hour'] = test2_time.hour
    df.loc[:, name + '_hour_cnt'] = df.groupby([name + '_index', 'hour'])[
        'hour'].transform('count')
    df.to_csv(path + 'test/test2_' + name + '_hour_cnt.csv')
    del df
    print name, 'finished'

# feature_half_hour_cnt

test2_time = pd.read_table(path + 'test_timestamps.csv', sep=',',
                           names=['timestamps'], skiprows=64814823)
test2_time['hour'] = pd.to_datetime(test2_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.hour
test2_time['mint'] = pd.to_datetime(test2_time.timestamps, unit='s').dt.tz_localize(
    'UTC').dt.tz_convert('Asia/Shanghai').dt.minute
test2_time['mint'] = (test2_time.mint / 30).astype(np.int)
test2_time['half_hour'] = test2_time.hour * 10 + test2_time.mint
names = ['cookie', 'f', 'placementid']

for name in names:
    df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[3])
    df.loc[:, 'half_hour'] = test2_time.half_hour
    df.loc[:, name + '_half_hour_cnt'] = df.groupby([name + '_index', 'half_hour'])[
        'half_hour'].transform('count')
    df.to_csv(path + 'test/test2_' + name + '_half_hour_cnt.csv')
    del df
    print name, 'finished'


# feature_minute_cnt

test2_minute = pd.read_csv(path + 'test/test2_' +
                           'time_minute.csv', usecols=['minute'])
names = [
    'cookie',
    'f',
    #          'idfa', 'android_id', 'mobile_openudid', 'mobile_mac',
    'placementid',
]

for name in names:
    df = pd.read_csv(path + 'test/test2_' + name + '.csv', usecols=[3])
    df.loc[:, 'minute'] = test2_minute.minute
    df.loc[:, name + '_minute_cnt'] = df.groupby([name + '_index', 'minute'])[
        'minute'].transform('count')
    df[name + '_minute_cnt_mean'] = df.groupby(
        [name + '_index'])[name + '_minute_cnt'].transform('mean')
    df[name + '_minute_cnt_std'] = df.groupby(
        [name + '_index'])[name + '_minute_cnt'].transform('std')
    df.to_csv(path + 'test/test2_' + name + '_minute_cnt.csv')
    del df
    print name, 'done!'

del test2_minute

# feature_minute_hour_cnt

names = [
    'cookie',
    'f',
    'placementid',
    'global_mediaid',
]
for name in names:
    df = pd.read_csv(path + 'test/test2_' + name + '_minute_cnt.csv',
                     usecols=[name + '_index', name + '_minute_cnt'])
    df['hour'] = pd.read_csv(path + 'test/test2_' +
                             'time_hour.csv', usecols=['hour'])
    df[name + '_minute_cnt_hour_mean'] = df.groupby([name + '_index', 'hour'])[
        name + '_minute_cnt'].transform('mean')
    df[name + '_minute_cnt_hour_std'] = df.groupby([name + '_index', 'hour'])[
        name + '_minute_cnt'].transform('std')
    df.to_csv(path + 'test/test2_' + name + '_hour_stat.csv',
              columns=[name + '_minute_cnt_hour_mean',
                       name + '_minute_cnt_hour_std'],
              index=False)
    del df
    print name, 'done!'


# media_info

media_dummy = pd.read_csv(path + 'media_dummy.csv')
test_media = pd.read_table(path + 'test_global_mediaid.csv', sep=',',
                           usecols=['global_mediaid'], skiprows=64814823)
test_media = test_media.merge(
    media_dummy, left_on='global_mediaid', right_on='id', how='left')
test_media.drop(['global_mediaid', 'id'], axis=1).to_csv(
    path + 'test/test2_media.csv', index=False)
