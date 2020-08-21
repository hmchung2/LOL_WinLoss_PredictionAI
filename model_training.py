import tensorflow as tf
import pandas as pd
import numpy as np
import time
from collections import Counter
from tqdm import tqdm
import pymysql
from sqlalchemy import create_engine, types, select
import api_config
import requests
#pymysql.install_as_MySQLdb()
#import MySQLdb
from sklearn.utils import shuffle
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import inspect
def inspect_hm(func):
    inspection = inspect.getargspec(func)
    args = inspection.args
    defaults =  np.array(inspection.defaults).tolist()
    num = int(len(args) - len(defaults))
    for i in range(num):
        defaults.insert(i , None)
    final_df = pd.DataFrame({"args":args , "defaults":defaults})
    return final_df




class connect_sql:
    def __init__(self,host,db_name,user_name,password,port, db_type):
        if db_type == "postgres":
            self.database = 'postgresql'
        elif db_type == "mysql":
            self.database = 'mysql+mysqldb'
        engine = create_engine(self.database +'://{}:{}@{}:{}/{}'.format(user_name, password, host,port , db_name))
        self.engine = engine
        self.conn = engine.connect()
    def insert_df(self, df, table_name):                                                   # 데이터프레임 통채로 데이터베이스에 삽입
        df.to_sql(table_name , con = self.engine, index = False, if_exists ='append', chunksize = 50 )
    def get_column_names(self,table_name):
        self.table_name = table_name
        query  = 'show columns from ' + self.table_name
        result = self.conn.execute(query).fetchall()
        self.columns = [x[0] for x in result]
    def get_data(self,columns,table_name):
        self.table_name = table_name
        query = 'select ' + columns + " from " + self.table_name
        result = self.conn.execute(query).fetchall()
        self.data = result
    def get_whole_data_frame(self, table_name):
        self.get_data("*",table_name)
        data = self.data
        self.get_column_names(table_name)
        columns = self.columns
        df = pd.DataFrame(data, columns = columns)
        self.df = df


def get_data(table_name):
    pymysql.install_as_MySQLdb()
    host = "192.168.0.181"
    db_name = "lolpred"
    user_name = "root"
    password = "123"
    port = 3306
    db_type = "mysql"
    connect_db = connect_sql(host,db_name,user_name,password,port, db_type)
    connect_db.get_whole_data_frame(table_name )
    df = connect_db.df
    connect_db.conn.close()
    return df



def modifiy_df(df):
    champ_index_booling = list(map(lambda x: "_100" not in x and "_200" not in x , df.columns.tolist() ))
    df = df.loc[:,champ_index_booling ]
    id_info_booling = list(map(lambda x: "sumName" not in x and "accountId" not in x and "sumID" not in x and "inactive" not in x , df.columns.tolist()  ) )
    df = df.loc[: , id_info_booling]
    return df


def get_current_version(key):
    api_key = key
    r = requests.get('https://ddragon.leagueoflegends.com/api/versions.json') # version data 확인
    current_version = r.json()[0]
    return current_version

def get_champion_id_by_current_version(key, version):
    api_key = key
    r = requests.get('http://ddragon.leagueoflegends.com/cdn/{}/data/ko_KR/champion.json'.format(version))
    parsed_data = r.json() # 파싱
    info_df = pd.DataFrame(parsed_data)
    champ_dic = {}
    for i, champ in enumerate(info_df.data):
        champ_dic[i] = pd.Series(champ)
    champ_df = pd.DataFrame(champ_dic).T
    champ_info_df = pd.DataFrame(dict(champ_df['info'])).T
    champ_stats_df = pd.DataFrame(dict(champ_df['stats'])).T


    # 데이터 합치기
    champ_df = pd.concat([champ_df, champ_info_df], axis=1)
    champ_df = pd.concat([champ_df, champ_stats_df], axis=1)
    # 이번 분석에서 필요없는 데이터 제거
    champ_df = champ_df.drop(['version', 'image', 'info', 'stats', 'blurb'], axis=1)
    return champ_df

def champion_key_to_id(champ_info,key_list):
    new_df = champ_info.set_index(pd.Series(champ_info.key.tolist()))
    champion_names = new_df.loc[key_list,'id']
    return champion_names

def switch_champId_to_champName(df , which_lanes, api_key):
    current_version = get_current_version(api_key)
    champ_info = get_champion_id_by_current_version(api_key , current_version)
    all_lanes = which_lanes.copy()
    for lane in all_lanes:
        champId_list = df["{}_champ".format(lane)].tolist()
        champId_list = list(map(lambda x: str(x) , champId_list  ))
        df["{}_champ".format(lane)] = champion_key_to_id(champ_info , champId_list).tolist()
    return df



def process_lastplaytime(df , which_lanes):
    modified_df = df.copy()
    all_lanes = which_lanes.copy()
    game_creation_array = np.array(modified_df.gameCreation.tolist())
    for lane in all_lanes:
        last_playtime_array = np.array( modified_df["{}_lastplaytime".format(lane)].tolist())
        avg_last_playtime_array = np.array(modified_df["{}_avg_lastplaytime".format(lane)].tolist())
        processed_lastplaytime =  game_creation_array  - last_playtime_array
        processed_avg_lastplaytime = game_creation_array - avg_last_playtime_array
        modified_df["{}_lastplaytime".format(lane)] = processed_lastplaytime.tolist()
        modified_df["{}_avg_lastplaytime".format(lane)] = processed_avg_lastplaytime.tolist()
    return modified_df



def del_last_playtime(df):
    bool_list = list(map(lambda x: "lastplaytime" not in x , df.columns.tolist()  ))
    df = df.loc[: , bool_list]
    return df




def change_roman_number(df , which_lanes):
    all_lanes = which_lanes.copy()
    modified_df3 = df.copy()
    subs = {"I":1 , "II":2 , "III":3 , "IV":4 }
    for lane in all_lanes:
        rank_list  =  modified_df3["{}_rank".format(lane)].tolist()
        modified_rank_list = list(map(lambda x: int(subs[x]) , rank_list  ))
        modified_df3["{}_rank".format(lane)] = modified_rank_list
        modified_df3["{}_rank".format(lane)] = modified_df3["{}_rank".format(lane)].astype(int)
    return modified_df3


def change_all_tiers(df , which_lanes):
    all_lanes = which_lanes.copy()
    modified_df3 = df.copy()
    subs = {"BRONZE":0 , "SILVER":1, "GOLD":2, "PLATINUM":3,"DIAMOND":4,"MASTER":5,"GRANDMASTER":6,"CHALLENGER":7}
    for lane in all_lanes:
        tier_list = modified_df3["{}_tier".format(lane)].tolist()
        modified_tier_list = list(map(lambda x: int(subs[x]) , tier_list  ))
        modified_df3["{}_tier".format(lane)] = modified_tier_list
        modified_df3["{}_tier".format(lane)] = modified_df3["{}_tier".format(lane)].astype(int)
    return modified_df3



# df = get_data("grandmaster_0807")
# df.to_csv("raw_data.csv")

df = pd.read_csv("raw_data.csv" , index_col = 0 )

df = modifiy_df(df)


main_api_key = api_config.main_api_key
all_lanes  = api_config.all_lanes


modified_df = switch_champId_to_champName(df, all_lanes , main_api_key)
modified_df2 =  process_lastplaytime(modified_df , all_lanes)
modified_df3 = del_last_playtime(modified_df2)     ### for now only....  will check this variable someday
modified_df5 = change_roman_number(modified_df3 , all_lanes)
modified_df5 = change_all_tiers(modified_df5 , all_lanes)


if "gameId" in modified_df5.columns.tolist(): del modified_df5["gameId"]
if "gameCreation" in modified_df5.columns.tolist(): del modified_df5["gameCreation"]

for col in modified_df5.columns.tolist():
    if col in modified_df5.select_dtypes(include=['float64']).columns.tolist():
        modified_df5[col] = modified_df5[col].astype(np.float32)



# fog project example ############################################
# feature_columns = []
# train_columns = train.columns.tolist()
# if "target" in train_columns: train_columns.remove("target")
# for header in train_columns:
#     feature_columns.append(feature_column.numeric_column(header))
#
# feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
##################################################################

# object  :tier ,
# true false : veteran  , inactive , freshBlood , hotStreak
# oridinal : rank

current_version = get_current_version(main_api_key)
champ_info = get_champion_id_by_current_version(main_api_key, current_version)


# for x, y in zip( modified_df5.columns, modified_df5.dtypes):
#     print( x , " : " ,  y)

# def create_feature_columns(df, champ_info):
#     modified_df5 = df.copy()
#     col_list = modified_df5.columns.tolist()
#     if "gameId" in col_list: col_list.remove("gameId")
#     if "gameCreation" in col_list: col_list.remove("gameCreation")
#     if "target" in col_list: col_list.remove("target")
#     tiers = api_config.tiers
#     feature_columns = []
#
#     for col in col_list:
#         if col.endswith("_champ") and "damage" not in col:
#             #print(col)
#             categ_var = feature_column.categorical_column_with_vocabulary_list(col , champ_info.id.tolist() )
#             the_one_hot = feature_column.indicator_column(categ_var)
#             feature_columns.append(the_one_hot)
#
#         elif col.endswith("_tier"):
#             categ_var = feature_column.categorical_column_with_vocabulary_list(col , tiers )
#             the_one_hot = feature_column.indicator_column(categ_var)
#             feature_columns.append(the_one_hot)
#         elif col.endswith("_veteran")  or col.endswith("_freshBlood") or col.endswith("_hotStreak"):
#             #print(col)
#             categ_var = feature_column.categorical_column_with_vocabulary_list(col , [0,1] )
#             the_one_hot = feature_column.indicator_column(categ_var)
#             feature_columns.append(the_one_hot)
#         else:
#             feature_columns.append(feature_column.numeric_column(col ))
#     return feature_columns
#
# feature_columns = create_feature_columns(modified_df5 , champ_info)
# feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

main_features = api_config.main_features
main_features.remove("target")

train, test = train_test_split(modified_df5, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)

feature_summary = train.describe()


def standardize(data , raw_summary):
    df = data.copy()
    summary = raw_summary.copy()
    label  = df.pop("target")
    summary.pop("target")
    for col in summary:
        normed_data = (df[col] - summary.loc["mean", col]) / summary.loc["std" , col]
        normed_data =  normed_data.tolist()
        df[col] = normed_data
    df["target"] = label
    return df

normed_train = standardize(train , feature_summary)
normed_test = standardize(test , feature_summary)
normed_val = standardize(val , feature_summary)

normed_train = normed_train.reset_index(drop = True)


def transform_df(raw_df):
    df = raw_df.copy()
    relations = api_config.feature_relations2
    all_lanes = api_config.all_lanes
    col_list = []
    for lane in all_lanes:
        col_list.append("{}_good_numeric".format(lane))
        col_list.append("{}_bad_numeric".format(lane))
        col_list.append("{}_good_categ".format(lane))
        col_list.append("{}_bad_categ".format(lane))
        col_list.append("{}_champ".format(lane))
    new_df = pd.DataFrame(columns = col_list)
    new_df = new_df.astype('object')
    for lane in all_lanes:
        new_df["{}_champ".format(lane)] = df["{}_champ".format(lane)]
        good_numeric_cols = list(map(lambda x: "{}_{}".format(lane , x) , relations["good"]  ))
        bad_numeric_cols = list(map(lambda x: "{}_{}".format(lane, x ) , relations["bad"]  ))
        good_categ_cols = list(map(lambda x: "{}_{}".format(lane, x) , relations["good_categ"]))
        bad_categ_cols = list(map(lambda x: "{}_{}".format(lane , x) , relations["bad_categ"] ))
        for index in df.index.tolist():
            new_df.loc[index ,"{}_good_numeric".format(lane)] = df.loc[index,good_numeric_cols].tolist()
            new_df.loc[index, "{}_bad_numeric".format(lane)] = df.loc[index, bad_numeric_cols].tolist()
            new_df.loc[index, "{}_good_categ".format(lane)] = df.loc[index, good_categ_cols].tolist()
            new_df.loc[index , "{}_bad_categ".format(lane)] = df.loc[index , bad_categ_cols].tolist()

    new_df["target"] = df["target"]
    return new_df


transformed_df = transform_df(normed_train)
transformed_df


labels = transformed_df.pop("target")
ds = tf.data.Dataset.from_tensor_slices((dict(transformed_df), labels  )  )
ds  = ds.shuffle(buffer_size  = len(transformed_df) )
ds  = ds.batch(5)
transformed_df
normed_train
transformed_df.columns

len(transformed_df.TOP100_good_categ.iloc[0])


feature_columns = create_feature_columns(transformed_df , all_lanes)
transformed_df.shape
len(feature_columns)


def create_feature_columns(df , which_lanes):
    transformed_df = df.copy()
    all_lanes = which_lanes.copy()
    champion_list = champ_info.id.tolist()
    feature_columns = []
    good_numeric = list(map(lambda x: "{}_good_numeric".format(x) , all_lanes ))
    for header in good_numeric:
        feature_columns.append(feature_column.numeric_column(header, shape = (22,) ))
    bad_numeric = list(map(lambda x: "{}_bad_numeric".format(x), all_lanes))
    for header in bad_numeric:
        feature_columns.append(feature_column.numeric_column(header , shape = (3, )))
    good_categ = list(map(lambda x: "{}_good_categ".format(x), all_lanes))
    for header in good_categ:
        feature_columns.append(feature_column.numeric_column(header , shape = (4,) ))
    bad_categ = list(map(lambda x: "{}_bad_categ".format(x), all_lanes))
    for header in bad_categ:
        feature_columns.append(feature_column.numeric_column(header , shape = (2,) ))
    TOP100 = feature_column.categorical_column_with_vocabulary_list("TOP100_champ" , champion_list)
    TOP200 = feature_column.categorical_column_with_vocabulary_list("TOP200_champ",  champion_list)
    TOP_crossed = feature_column.crossed_column([TOP100 ,TOP200] , hash_bucket_size = 1000)
    TOP_crossed = feature_column.indicator_column(TOP_crossed)
    feature_columns.append(TOP_crossed)

    JUNGLE100 = feature_column.categorical_column_with_vocabulary_list("JUNGLE100_champ" , champion_list)
    JUNGLE200 = feature_column.categorical_column_with_vocabulary_list("JUNGLE200_champ" , champion_list)
    JUNGLE_crossed = feature_column.crossed_column([JUNGLE100, JUNGLE200] , hash_bucket_size = 1000)
    JUNGLE_crossed = feature_column.indicator_column(JUNGLE_crossed)
    feature_columns.append(JUNGLE_crossed)

    MID100 = feature_column.categorical_column_with_vocabulary_list("MID100_champ" , champion_list)
    MID200 = feature_column.categorical_column_with_vocabulary_list("MID200_champ" , champion_list)
    MID_crossed = feature_column.crossed_column([MID100,MID200] , hash_bucket_size = 1000)
    MID_crossed = feature_column.indicator_column(MID_crossed)
    feature_columns.append(MID_crossed)

    ADC100 = feature_column.categorical_column_with_vocabulary_list("ADC100_champ" , champion_list)
    ADC200 = feature_column.categorical_column_with_vocabulary_list("ADC200_champ" , champion_list)
    ADC_crossed = feature_column.crossed_column([ADC100 , ADC200] , hash_bucket_size = 1000)
    ADC_crossed = feature_column.indicator_column(ADC_crossed)
    feature_columns.append(ADC_crossed)

    SUPPORT100 = feature_column.categorical_column_with_vocabulary_list("SUPPORT100_champ" , champion_list)
    SUPPORT200 = feature_column.categorical_column_with_vocabulary_list("SUPPORT200_champ" , champion_list)
    SUPPORT_crossed = feature_column.crossed_column([SUPPORT100 , SUPPORT200] , hash_bucket_size = 1000)
    SUPPORT_crossed = feature_column.indicator_column(SUPPORT_crossed)
    feature_columns.append(SUPPORT_crossed)

    return feature_columns


feature_columns = create_feature_columns(transformed_df , all_lanes)
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

train_ds = df_to_dataset(transformed_df)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

feature_columns
transformed_df
normed_train

#tf.keras.backend.set_floatx('float64')
model.fit(train_ds, validation_data=train_ds, epochs=5)




def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds





for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))


def transform_df(df):
    lanes = api_config.all_lanes
    feature_relations = api_config.feature_relations
    normed_train = df.copy()
    new_df = pd.DataFrame()
    for index in df:
        for lane in all_lanes:
            good = []
            bad = []
            good_categ = []
            bad_categ = []
            for key, value in feature_relations.items():
                if value == 'good':
                    good.append(normed_train.loc[index, "{}_{}".format(lane, key)] )
                elif value == 'bad':
                    bad.append(normed_train.loc[index ])


Counter(feature_relations.values())
df = normed_train.iloc[:2 , :]
df = df.reset_index(drop = True)

new_df = pd.DataFrame({"no_data":[[1,2,3]]})



normed_train["TOP100_tier"]

train_ds = df_to_dataset(normed_train, batch_size=batch_size)
val_ds = df_to_dataset(normed_val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(normed_test, shuffle=False, batch_size=batch_size)







model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)




test = list(range(30))
first = list(map(lambda x: "rock" if x < 15 else "paper" , test ))
second =list(map(lambda x: "paper" if x  == "rock"  else "rock" , first ))
target = list(map(lambda x: 0 if x  == "rock"  else 1 , first ))

test_df = pd.DataFrame({"first":first ,"second" : second , "target" : target} )

df = shuffle(test_df)
df = df.reset_index(drop = True)

labels = df.pop("target")
ds = tf.data.Dataset.from_tensor_slices((dict(df), labels  )  )
ds  = ds.shuffle(buffer_size  = len(df) )
ds  = ds.batch(5)

thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)




feature_columns = []

first  =  feature_column.categorical_column_with_vocabulary_list("first" , ["rock" , "paper"])
first_one_hot = feature_column.indicator_column(first)
feature_columns.append(first_one_hot)

second  =  feature_column.categorical_column_with_vocabulary_list("second" , ["rock" , "paper"])
second_one_hot = feature_column.indicator_column(second)
feature_columns.append(second_one_hot)

crossed_feature = feature_column.crossed_column([first, second], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)

inspect.getargspec(crossed_feature.create_state)






inspect(crossed_feature.index)
dir(crossed_feature)

feature_columns.append(crossed_feature)


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)





model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#tf.keras.backend.set_floatx('float64')
model.fit(ds, validation_data=ds, epochs=100)


check_df = pd.DataFrame({"first":["paper","paper","rock","paper"] , "second":["rock" , "rock" , "paper", "paper"] })
check_df["target"] = 0

check_label = check_df.pop("target")
check_ds = tf.data.Dataset.from_tensor_slices((dict(check_df), check_label  )  )
check_ds = check_ds.batch(1)

model.predict(check_ds, batch_size = None)




def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds






train, test = train_test_split(modified_df5, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)




# def standardize(data , summary):
#     df = data.copy()
#     df = df.reset_index()
#     df.reindex()
#     new_df = pd.DataFrame()
#     for col in summary:
#         normed_data = (df[col] - summary.loc["mean", col]) / summary.loc["std" , col]
#         normed_data =  normed_data.tolist()
#         new_df[col] = normed_data
#     new_df["target"] = df["target"].tolist()
#     return new_df
#




normed_train = standardize(train , feature_summary)
normed_test = standardize(test , feature_summary)
normed_val = standardize(val , feature_summary)

# training_columns = normed_train.columns.tolist()
# training_columns.remove("target")
# feature_columns = []
# for col in training_columns : feature_columns.append(feature_column.numeric_column(col ))

# feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# feature_columns


batch_size = 5 # 예제를 위해 작은 배치 크기를 사용합니다.

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds



train_ds = df_to_dataset(normed_train, batch_size=batch_size)
val_ds = df_to_dataset(normed_val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(normed_test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])




#tf.keras.backend.set_floatx('float64')


model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

# #feature_columns = []
#
# # 수치형 열
# for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
#   feature_columns.append(feature_column.numeric_column(header))
# test = feature_column.numeric_column("test")


# # 버킷형 열
# age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
# feature_columns.append(age_buckets)
#
# # 범주형 열
# thal = feature_column.categorical_column_with_vocabulary_list(
#       'thal2', ['fixed', 'normal', 'reversible'])

# thal_one_hot = feature_column.indicator_column(thal)
# feature_columns.append(thal_one_hot)
#
# # 임베딩 열
# thal_embedding = feature_column.embedding_column(thal, dimension=8)
# feature_columns.append(thal_embedding)
#
# # 교차 특성 열
# crossed_feature = feature_column.crossed_column([test, thal], hash_bucket_size= 10)

# crossed_feature = feature_column.indicator_column(crossed_feature)
# feature_columns.append(crossed_feature)
# feature_layer = tf.keras.layers.DenseFeatures(feature_columns)



champ_info


URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), '훈련 샘플')
print(len(val), '검증 샘플')
print(len(test), '테스트 샘플')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

feature_columns = []

# 수치형 열
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

# 버킷형 열
age = feature_column.numeric_column("age")
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 범주형 열
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# 임베딩 열
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# 교차 특성 열
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)
