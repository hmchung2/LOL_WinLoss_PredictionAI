import tensorflow as tf
import pandas as pd
import numpy as np
import time
from collections import Counter
from tqdm import tqdm
import pymysql
from sqlalchemy import create_engine, types, select
#pymysql.install_as_MySQLdb()
#import MySQLdb



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


def get_data():
    pymysql.install_as_MySQLdb()
    host = "192.168.0.181"
    db_name = "lolpred"
    user_name = "root"
    password = "123"
    port = 3306
    db_type = "mysql"
    connect_db = connect_sql(host,db_name,user_name,password,port, db_type)
    connect_db.get_whole_data_frame("grandmaster_0807")
    df = connect_db.df
    connect_db.conn.close()
    return df

def main():
    return True
    
df = get_data()
def modifiy_df(df):
    del


champ_index_booling = list(map(lambda x: "_100" not in x and "_200" not in x , df.columns.tolist() ))
df = df.loc[:,champ_index_booling ]
id_info_booling = list(map(lambda x: "sumName" not in x and "accountId" not in x and "sumID" not in x , df.columns.tolist()  ) )
df = df.loc[: , id_info_booling]

for i in df.columns:
    print(i ," : " ,df[i].iloc[0] )
