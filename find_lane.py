import pickle # 리스트 안의 데이터프레임 형태 저장
import requests # api 요청
import json
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter
from tqdm import tqdm
from skimage import io # 미니맵 처리
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_numeric_dtype
from statistics import mean
import api_config
from sqlalchemy import create_engine, types, select
from sqlalchemy import *
import pymysql
from packaging import version

pymysql.install_as_MySQLdb()


import pandas as pd
from sqlalchemy import create_engine

# MySQL Connector using pymysql
import MySQLdb

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



############################################ not interchangeable ####### random evertime#################################
def show_grandmaster_info(api_key):
    grandmaster = 'https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key=' + api_key
    r = requests.get(grandmaster)
    league_df = pd.DataFrame(r.json())
    league_df.reset_index(inplace=True)
    league_entries_df = pd.DataFrame(dict(league_df['entries'])).T
    league_df = pd.concat([league_df, league_entries_df], axis=1)
    league_df = league_df.drop(['index', 'queue', 'name', 'leagueId', 'entries', 'rank'], axis=1)
    return league_df

def show_info(api_key, tier, division, page = 1):
    if tier.isupper() != True:
        raise ValueError('please write tier in upper case')
    roman_num = {'1':'I' , '2':'II', '3':'III', '4':'IV'}
    division_roman = roman_num[str(division)]
    query = 'https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/'+ tier+ '/'+ division_roman + '?page='+ str(page) +'&api_key=' + api_key
    r = requests.get(query)
    league_df = pd.DataFrame(r.json())
    return league_df



#leagueId = show_info(api_key,'GOLD',1, 2)["leagueId"][0]

#######################3 not interchangeable #############################
def df_summoner_accountid(league_df,api_key):
    league_df['account_id'] = None
    for i in range(len(league_df)):
        try:
            sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key
            r = requests.get(sohwan)

            while r.status_code == 429 or r.status_code == 504:
                time.sleep(5)
                #print('time to wait')
                sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key
                r = requests.get(sohwan)

            account_id = r.json()['accountId']

            league_df.iloc[i, -1] = account_id
            #print('going good')
        except:
            #print('not ok')
            pass
    #print('done')
    return league_df



#
############################## not interchangeable ##########################################
def accountID_to_matchINFO(league_df3, endIndex , api_key):
    # need account_id column in the data frame
    match_info_df = pd.DataFrame()
    season = str(13)
    EI = str(endIndex)
    len(league_df3)
    for i in range(len(league_df3)):
        try:
            match0 = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + league_df3['account_id'].iloc[i]  +'?season=' + season +'&endIndex='+ EI+'&api_key=' + api_key
            r = requests.get(match0)

            while r.status_code == 429:
                time.sleep(5)
                #print('time to wait')
                match0 = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + league_df3['account_id'].iloc[i]  +'?season=' + season +'&endIndex='+ EI+'&api_key=' + api_key
                r = requests.get(match0)

            match_info_df = pd.concat([match_info_df, pd.DataFrame(r.json()['matches'])])
            #print('going good')
        except:
            continue
            #print('not going good')
            #print(i)
    print("done")
    return match_info_df


# Match 데이터 받기 (gameId를 통해 경기의 승패, 팀원과 같은 정보가 담겨있다.)
################# interchangeable but not recommending since we will use the data in the futrue  #########################33
def game_id_to_match_detail(match_info_df2, api_key):
    match_fin = pd.DataFrame()
    for i in range(len(match_info_df2)):
        try:
            api_url='https://kr.api.riotgames.com/lol/match/v4/matches/' + str(match_info_df2['gameId'].iloc[i]) + '?api_key=' + api_key
            r = requests.get(api_url)

            while r.status_code == 429 or r.status_code == 504:
                time.sleep(2)
                #print('time to wait')
                #time.sleep를 꼭 해줘야함 안그러면 request 잦은 사용으로 블랙리스트가 됨
                api_url='https://kr.api.riotgames.com/lol/match/v4/matches/' + str(match_info_df2['gameId'].iloc[i]) + '?api_key=' + api_key
                r = requests.get(api_url)

            mat = pd.DataFrame(list(r.json().values()), index=list(r.json().keys())).T
            match_fin = pd.concat([match_fin,mat])
            #print('going well')
        except:
            #print('not going well')
            pass
    match_fin2 = match_fin.reset_index()
    del match_fin2['index']
    #print('done')
    return match_fin2



def find_new_version(df):
    match_df = df.copy()
    version_list = match_df["gameVersion"].unique().tolist()
    version_list = [x  for x in version_list if type(x) == type('string')]

    for n, i in enumerate(version_list):
        version_list[n] = version.parse(i)
    result = max(version_list)
    return result.public



# when running in module withou stopping
def modifiy_match_df_original(df):
    match_df = df.copy()
    #del match_df["status"]
    match_df = match_df.dropna()
    match_df["gameId"] = list(map(lambda x: int(x), match_df["gameId"]))
    match_df["queueId"] = match_df["queueId"].map(lambda x : int(x))
    match_df["gameCreation"] = match_df["gameCreation"].map(lambda x : int(x))
    match_df["seasonId"] = match_df["seasonId"].map(lambda x : int(x))
    match_df["mapId"] = match_df["mapId"].map(lambda x : int(x))
    match_df["gameDuration"] = match_df["gameDuration"].map(lambda x : int(x))
    new_version = find_new_version(match_df)
    match_df = match_df.loc[(match_df['gameVersion']==new_version) & (match_df['gameMode']=='CLASSIC'), :]
    select_indices = (match_df['queueId']==420) | (match_df['queueId']==440)
    match_df = match_df.loc[select_indices, :].reset_index(drop=True)
    return match_df

# def modfify_match_df(df):
#     match_df = df.copy()
#     match_df["gameId"] = list(map(lambda x: int(x), match_df["gameId"]))
#     match_df["queueId"] = match_df["queueId"].map(lambda x : int(x))
#     match_df["gameCreation"] = match_df["gameCreation"].map(lambda x : int(x))
#     match_df["seasonId"] = match_df["seasonId"].map(lambda x : int(x))
#     match_df["mapId"] = match_df["mapId"].map(lambda x : int(x))
#     match_df["gameDuration"] = match_df["gameDuration"].map(lambda x : int(x))
#
#     new_version = find_new_version(match_df)
#
#     # 정확한 통계를 위해 가장 최신의 버전과 클래식 게임에 대한 데이터만 가져오자
#     match_df = match_df.loc[(match_df['gameVersion']==new_version) & (match_df['gameMode']=='CLASSIC'), :]
#
#     # 그 중에서도 이번 분석에서는 소환사의 협곡 솔로 랭크와 팀 랭크 게임만 사용한다.
#     select_indices = (match_df['queueId']==420) | (match_df['queueId']==440)
#
#     match_df = match_df.loc[select_indices, :].reset_index(drop=True)
#
#     # DataFrame 내의 리스트들이 파일로 저장되었다가 불러지는 과정에서 문자로 인식됨
#     for column in ['teams', 'participants', 'participantIdentities']:
#         match_df[column] = match_df[column].map(lambda v: eval(v)) # 각 값에 대해 eval 함수를 적용
#     return match_df


class api_box:
    def __init__(self, api_key_list):
        self.number = 0
        self.length_list  = len(api_key_list)
        self.api_key_list = api_key_list
        self.current_api_key = api_key_list[0]
    def switch(self):
        self.number = self.number +1
        new_index = self.number % self.length_list
        self.current_api_key = self.api_key_list[new_index]
        return self.current_api_key
    def returning(self):
        return self.current_api_key


# this one works with different apis
# 각 타임라인에 찍힌 위치 정보가 필요한데, match-timelines 데이터에 모여있다.
# 그래서 이 데이터를 가져와야한다.
def get_time_line_list(df,main_api_key, api_key_list):
    match_df = df.copy()
    match_timeline_list = []
    api_key_list.append(main_api_key)
    api_machine = api_box(api_key_list)
    for game_id in tqdm(match_df['gameId']): # 각 게임 아이디마다 요청
        api_url = 'https://kr.api.riotgames.com/lol/match/v4/timelines/by-match/{}?api_key={}'.format(game_id , api_machine.current_api_key)
        r = requests.get(api_url)
        count = 0
        while r.status_code == 429 or r.status_code == 504:
            if r.status_code == 504:
                print("gateaway timeout")
                count = count + 1
                if count == 150:
                    break

             # 요청 제한 또는 오류로 인해 정상적으로 받아오지 않는 상태라면, 3초 간 시간을 지연
            time.sleep(3)
            api_url = 'https://kr.api.riotgames.com/lol/match/v4/timelines/by-match/{}?api_key={}'.format(game_id , api_machine.switch())
            r = requests.get(api_url)
        temp_match = pd.DataFrame(list(r.json().values())[0]) # 전체 데이터 저장 (데이터 값에 딕셔너리 형태로 필요한 정보가 저장)
        temp_timeline = pd.DataFrame()
        len_timeline = temp_match.shape[0]
        for i in range(len_timeline): # 각 게임의 타임라인이 모두 다르기 때문 (1분 가량마다 타임라인이 찍힌다)
            temp_current_timeline = pd.DataFrame(temp_match['participantFrames'].iloc[i]).T
            if i != (len_timeline-1):
                temp_position = pd.DataFrame(list(temp_current_timeline['position'].values), index=temp_current_timeline.index)
                temp_current_timeline = pd.concat([temp_current_timeline, temp_position], axis=1)
                temp_current_timeline.drop('position', axis=1, inplace=True)
            temp_current_timeline['timestamp'] = temp_match['timestamp'][i]
            temp_timeline = pd.concat([temp_timeline, temp_current_timeline], axis=0, sort=False)
        temp_timeline["gameId"] = game_id
        match_timeline_list.append(temp_timeline)
    return match_timeline_list


def participants_for_lanes(match, timeline):
    match_df = match.copy()
    match_timeline_list = timeline.copy()
    lane_calculated = pd.DataFrame()
    for k in tqdm(range(len(match_timeline_list)  ) ):              #len(match_timeline_list)
        game_id = match_timeline_list[k]["gameId"][0]
        if match_df.loc[match_df["gameId"] == game_id , "gameDuration"].tolist()[0] < 600: continue
        cur_timeline = match_timeline_list[k].copy()                  ######## will modify
        cur_timeline['jungleMinionsKilled'] = cur_timeline['jungleMinionsKilled'].astype('float64')
        cur_timeline['minionsKilled'] = cur_timeline['minionsKilled'].astype('float64')
        # 타임스탬프는 op.gg가 나타내는 아이템 타임스탬프와 비교 결과, 타임스탬프 값의 1000 단위가 1초인 것을 파악함
        cur_timeline['timestamp'] = cur_timeline['timestamp'] / (1000*60)  # 타임스탬프 값을 분 단위로 변환
        condition = (cur_timeline['timestamp'] > 2) & (cur_timeline['timestamp'] < 15)     # 15분 이상 : 라인전을 끝내고 다른 라인으로 이동할 수 있음
        cur_timeline = cur_timeline.loc[condition].copy()
        cur_timeline['x'], cur_timeline['y'] = MapScaler(cur_timeline)
        player_spells = [(data['spell1Id'], data['spell2Id'])for data in match_df.loc[match_df["gameId"] == game_id, "participants" ].tolist()[0] ]  # 스펠 확인
        player_spells = np.array(player_spells)
        lane = {}
        team = 100
        for i in range(1, 11, 5):  # 라인 계산
            jungle_participant,support_participant,jungle_index,support_index = SupJugPredict(cur_timeline, player_spells, i)
            lane["JUNGLE_{}".format(str(team) )] = jungle_participant
            for j in range(i, i+5):
                if str(j) == jungle_index:
                    continue
                cur_player = cur_timeline.loc[str(j)].copy()
                tempo_lane, _ = LanePredict(cur_player, support_index==str(j), jungle_index)
                lane[tempo_lane+"_{}".format(str(team))] = cur_player["participantId"].tolist()[0]
            team = team + 100

        lane['gameId'] = game_id
        lane = pd.Series(lane)
        if lane.value_counts().max() > 1: # 각 게임에 한 라인이 2명 이상이면 계산 착오로 판단하여 데이터 삭제
            print("{}번째 계산이 잘못되었습니다.".format(str(match_df["gameID"][k] ) ))
            continue
        lane_calculated = pd.concat([lane_calculated, pd.Series(lane)], axis=1, sort=False)
    return lane_calculated.T


# f = open('MatchTimelineData.pickle', 'wb') # 리스트 안의 데이터프레임 형태이므로 바이너리 코드로 저장하기 위함임
# pickle.dump(match_time_list, f)
# f.close()

# 블랙리스트 되서 또 11788개의 데이터만 받아왔음

# f = open('MatchTimelineData.pickle', 'rb')
# match_timeline_list = pickle.load(f)




def spell_general_info():
    spell_api = 'http://ddragon.leagueoflegends.com/cdn/9.3.1/data/ko_KR/summoner.json'
    r = requests.get(spell_api)
    spell_info_df = pd.DataFrame(r.json())
    spell = {}
    for i in range(len(spell_info_df)):
        cur_spell = spell_info_df['data'].iloc[i]
        if 'CLASSIC' in cur_spell['modes']:
            spell[int(cur_spell['key'])] = cur_spell['name']
    spell = sorted(spell.items(), key=lambda t : t[0])
    return spell



def MapScaler(data, x_range=(-120, 14870), y_range=(-120, 14980)): # x, y의 범위
    x = data['x'].astype('float64').values.reshape(-1, 1)
    y = data['y'].astype('float64').values.reshape(-1, 1)
    x_range = np.array(x_range).astype('float64').reshape(-1, 1)
    y_range = np.array(y_range).astype('float64').reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 512)) # 0~512로 변환
    scaler.fit(x_range)
    x = scaler.transform(x)
    scaler.fit(y_range)
    y = scaler.transform(y)
    return x, 512 - y


def SupJugPredict(data, player_spells, i): # 서폿과 정글을 판단하기 위한 함수
    if i==1: # 블루팀
        final_timeline = data.iloc[-10:-5].copy()
        spells = player_spells[:5]
    if i==6: # 레드팀
        final_timeline = data.iloc[-5:].copy()
        spells = player_spells[5:10]
    smite_count = (spells == 11).sum()
    # 각 팀의 스마이트 개수 (스마이트가 2명 이상이 보유하면, 라인의 위치로 판단하기 위함)
    if smite_count != 1:
        jungle_index = final_timeline['jungleMinionsKilled'].idxmax()
        jungle_participant = final_timeline.loc[jungle_index, "participantId"]
        final_timeline = final_timeline.drop(index=jungle_index)

    #elif  smite_count > 1: jungle_index = False
    else:
        jungle_participant = np.where(spells == 11)[0][0] + i
        jungle_index = final_timeline.loc[final_timeline["participantId"] == jungle_participant].index[0]
        final_timeline = final_timeline.loc[final_timeline["participantId"] != jungle_participant,:]
    support_index =  final_timeline['minionsKilled'].idxmin()
    support_participant = final_timeline.loc[support_index , "participantId"  ]

    return jungle_participant ,support_participant,jungle_index,support_index # 정글을 제외하고 cs가 가장 적은 사람이 서포터



def LanePredict(data, support_bool=False, jungle_bool=True):
    lane = {'TOP': 0, 'MID': 0, 'ADC': 0, 'JUNGLE': 0}
    etc = 0
    for xi, yi in zip(data['x'], data['y']):
        if (xi > 20) & (xi < 60) & (yi > 30) & (yi < 220): lane['TOP'] += 1
        elif (xi > 20) & (xi < 150) & (yi > 20) & (yi < 160): lane['TOP'] += 1
        elif (xi > 30) & (xi < 200) & (yi > 20) & (yi < 60): lane['TOP'] += 1
        elif (xi > 195) & (xi < 265) & (yi > 250) & (yi < 310): lane['MID'] += 1
        elif (xi > 220) & (xi < 295) & (yi > 220) & (yi < 290): lane['MID'] += 1
        elif (xi > 250) & (xi < 320) & (yi > 200) & (yi < 260): lane['MID'] += 1
        elif (xi > 290) & (xi < 350) & (yi > 160) & (yi < 215): lane['MID'] += 1
        elif (xi > 160) & (xi < 220) & (yi > 290) & (yi < 340): lane['MID'] += 1
        elif (xi > 310) & (xi < 460) & (yi > 435) & (yi < 485): lane['ADC'] += 1
        elif (xi > 400) & (xi < 490) & (yi > 385) & (yi < 480): lane['ADC'] += 1
        elif (xi > 440) & (xi < 490) & (yi > 310) & (yi < 455): lane['ADC'] += 1
        elif (xi > 0) & (xi < 170) & (yi > 340) & (yi < 512): etc += 1
        elif (xi > 340) & (xi < 512) & (yi > 0) & (yi < 170): etc += 1
        else: lane['JUNGLE'] += 1
    if jungle_bool:
        del lane['JUNGLE']
    # 예측된 서포터 번호가 봇에 가장 오래 있었으면 서포터로 확정, 아니면 라인으로 판단
    if support_bool & (max(lane, key=lane.get) == 'ADC'): return 'SUPPORT', lane
    return max(lane, key=lane.get), lane






# 각 게임 마다 누가 어느 라인을 맡았는지 알아주는 데이터프레임 생성


def my_load_match_df():
    match_df = pd.read_csv("match_df.csv", index_col = 0)
    for column in ['teams', 'participants', 'participantIdentities']:
        match_df[column] = match_df[column].map(lambda v: eval(v))
    return match_df








######################## -----                 start from here  ----------------------  ##########################################
#lane_matching_df.to_csv("lane_matching.csv")
#lane_matching_df = pd.read_csv("lane_matching.csv", index_col = 0)
#match_df = my_load_match_df()

# f = open('MatchTimelineData.pickle', 'rb')
# match_timeline_list = pickle.load(f)
# f.close()
#df[df.isnull().any(axis=1)]
# lane_matching_df[lane_matching_df.isnull().any(axis=1)]



def modify_lane_matching_df(df):
    lane_matching_df = df.copy()
    lane_matching_df = lane_matching_df.reset_index(drop = True)
    lane_matching_df = lane_matching_df.dropna()
    lane_matching_df = lane_matching_df.astype(int)
    return lane_matching_df

# 널 값 처리 및 전부 정수로 변환

# 라인 매칭 데이터 프레임과 게임 정보 데이터프레임 합치고 데이터프레임 생성
def merge_lane_info_to_match_info(match, lane):
    match_info = match.copy()
    lane_info = lane.copy()
    merged = pd.merge(match_info,lane_info, on = "gameId" , how ="inner")
    return merged


# 필요 없는 변수 삭제 함수
def modify_merged_info(df):
    merged_info =df.copy()
    del merged_info["platformId"]
    del merged_info["gameDuration"]
    del merged_info["queueId"]
    del merged_info["mapId"]
    del merged_info["seasonId"]
    del merged_info["gameVersion"]
    del merged_info["gameMode"]
    del merged_info["gameType"]
    #del merged_info["status"]
    return merged_info


# 승패 가리는 변수 생성후 target 컬럼으로 저장
def get_win_loss_col(df):
    merged_info = df.copy()
    if sum(list(map(lambda x: x[0]["teamId"] == 100, merged_info.teams ))) ==merged_info.shape[0]:
        data = list(map(lambda x: x[0]["win"] , merged_info.teams ))
        data = list(map(lambda x: 1 if x == "Win" else 0, data ))
        merged_info["target"] = data
        return merged_info
    else:
        print("manually throwing an error")
        #5/0




# 챔피온 아이디 플러스 서머너 encrypted
def get_champion_sumId_cols(df):
    merged_info = df.copy()
    # 먼저 챔피언 종류 필터
    tempo_list = ["TOP","JUNGLE","MID","ADC","SUPPORT"]
    for team in ["100","200"]:
        for lane in tempo_list:
            tempo_champions  = list(map(lambda x,y : y[int(x-1)]['championId'] if y[int(x-1)]['participantId'] == int(x) else np.nan, merged_info["{}_{}".format(lane, team)], merged_info["participants"]))
            tempo_summonerId = list(map(lambda x,y : y[int(x-1)]["player"]['summonerId'] if y[int(x-1)]['participantId'] == int(x) else np.nan, merged_info["{}_{}".format(lane, team)], merged_info["participantIdentities"]  ))
            tempo_accountId = list(map(lambda x, y : y[int(x-1)]["player"]["accountId"] if y[int(x-1)]['participantId'] == int(x)  else np.nan, merged_info["{}_{}".format(lane, team)], merged_info["participantIdentities"] ))
            tempo_sumName = list(map(lambda x, y : y[int(x-1)]["player"]["summonerName"] if y[int(x-1)]['participantId'] == int(x)  else np.nan, merged_info["{}_{}".format(lane, team)], merged_info["participantIdentities"] ))
            merged_info["{}{}_champ".format(lane,team)] = tempo_champions
            merged_info["{}{}_sumID".format(lane,team)] = tempo_summonerId
            merged_info["{}{}_accountId".format(lane,team)] = tempo_accountId
            merged_info["{}{}_sumName".format(lane,team)] = tempo_sumName
    return merged_info



# def champion_detail_for_all_lanes(df,api_key_list,lanes):
#     #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
#     merged_added = df.copy()
#     for lane in all_lanes:
#         merged_added = get_champion_detail_info(merged_added, api_key_list,lane)
#     return merged_added
#
#
#
#
# def get_champion_detail_info(df,api_key_list, lane_team):
#     merged_added = df.copy()
#     api_key_first = api_key_list[0]
#     api_key_second = api_key_list[1]
#     path ='https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{}/by-champion/{}?api_key={}'
#     merged_added["{}_champion_levels".format(lane_team)] = None
#     merged_added["{}_champion_points".format(lane_team)] = None
#     merged_added["{}_tokens".format(lane_team)] = None
#     merged_added["{}_lastplaytime".format(lane_team)] = None
#     api_urls_first = list(map(lambda sum, cham: path.format(sum,cham,api_key_first) , merged_added["{}_sumID".format(lane_team)] , merged_added["{}_champ".format(lane_team)]))
#     api_urls_second = list(map(lambda sum, cham: path.format(sum,cham,api_key_first) , merged_added["{}_sumID".format(lane_team)] , merged_added["{}_champ".format(lane_team)]))
#     if len(api_urls_first) != len(merged_added) or len(api_urls_second) != len(merged_added):
#         print("created api urls do not have the same length as the df, aborting and causing an error on purpose")
#         5 / 0
#     switching = 1
#     for i in tqdm(range(  len(api_urls_first ) )):
#         time.sleep(1)
#         api_url_first = api_urls_first[i]
#         api_url_second = api_urls_second[i]
#         try:
#             if switching % 2 == 0:
#                 api_url = api_url_seocnd
#             else:
#                 api_url = api_url_first
#             r = requests.get(api_url)
#             while r.status_code == 429:
#                 time.sleep(1)
#                 swithcing = switching + 1
#                 if switching % 2 == 0 :
#                     api_url = api_url_second
#                 else:
#                     api_url = api_url_first
#                 r = requests.get(api_url)
#             current_json = r.json()
#             merged_added["{}_champion_levels".format(lane_team)].iloc[i] = current_json["championLevel"]
#             merged_added["{}_champion_points".format(lane_team)].iloc[i] = current_json["championPoints"]
#             merged_added["{}_tokens".format(lane_team)].iloc[i] = current_json["tokensEarned"]
#             merged_added["{}_lastplaytime".format(lane_team)].iloc[i] = current_json["lastPlayTime"]
#         except:
#             print("an unknown error occured" )
#     return merged_added




#####################3



def coerce_df_columns_to_numeric(df):
    columns = []
    for column in df.columns.tolist():
        if is_numeric_dtype(df[column]):
            columns.append(column)
    column_list = columns
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
    return df

def del_nan_merge_large(df):
    merged_large = df.copy()
    merged_large = merged_large.dropna()
    merged_large = coerce_df_columns_to_numeric(merged_large)
    return merged_large


# failed just sum of levels in two
#one = path +"lol/champion-mastery/v4/scores/by-summoner/{}?api_key={}".format(encryptedSummonerId,api_key)



# every champion detail important
################ done ###############
#two = path +"lol/champion-mastery/v4/champion-masteries/by-summoner/{}?api_key={}".format(encryptedSummonerId,api_key)
#####################################



# 티어 승수 important
##################### done ##########################33
#three = path + "lol/league/v4/entries/by-summoner/{}?api_key={}".format(encryptedSummonerId,api_key)
############################################################
# failed just bunch of many summoners in the same league id
#four = path +"lol/league/v4/leagues/{}?api_key={}".format(gold_1.leagueId.iloc[0],api_key)

# summoner level and there is also puiid i dont know how to use might be important

######################### done ################################
#five = path +"lol/summoner/v4/summoners/{}?api_key={}".format(encryptedSummonerId,api_key)
#################################################################

# same as 5 but just use name instead
#six = path +"lol/summoner/v4/summoners/by-name/{}?api_key={}".format(summonerName,api_key)

def get_summonerLevel(df, main_api_key, api_key_listing ,lane_team):
    merged_large = df.copy()
    api_key_list = api_key_listing.copy()
    api_key_list.append(main_api_key)
    api_machine = api_box(api_key_list)
    path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-account/{}?api_key={}"
    temp_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
    merged_large["{}_summonerLevel".format(lane_team)] = None
    for i in tqdm(range(len(merged_large  ))):
        try:
            #time.sleep(1)
            api_url = path.format(merged_large["{}_accountId".format(lane_team)].iloc[i] , main_api_key )
            r = requests.get(api_url)
            trying = True
            count = 0
            while r.status_code == 429 or r.status_code ==504:
                if trying:
                    try:
                        if r.status_code == 504:
                            print("gateaway timeout")
                            count = count + 1
                            if count == 150:
                                break
                        api_url = temp_path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] ,  api_machine.switch() )
                        r = requests.get(api_url)
                        if r.status_code == 404:
                            api_url = path.format(merged_large["{}_accountId".format(lane_team)].iloc[i] , main_api_key )
                            trying = False
                            r = requests.get(api_url)
                    except Exception as e:
                        print("an error_here {}".format(e))
                        trying = False
                        continue
                else:
                    time.sleep(1)
                    api_url = path.format(merged_large["{}_accountId".format(lane_team)].iloc[i] , main_api_key )
                    r = requests.get(api_url)





            merged_large["{}_summonerLevel".format(lane_team)].iloc[i] = r.json()["summonerLevel"]

        except Exception as e:
                print("an error occured {}".format(e))
                print(i)
    return merged_large


def get_summonerLevel_for_all_lanes(df, main_api_key, api_key_listing ,all_lanes):
    merged_added = df.copy()
    api_key_list = api_key_listing.copy()
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    for lane in all_lanes:
        print(lane)
        merged_added = get_summonerLevel(merged_added, main_api_key, api_key_list ,lane)
    return merged_added




def get_win_los_rate_info(df, main_api_key, api_key_list, lane_team):
    merged_large = df.copy()
    api_machine  = api_box(api_key_list)
    path = "https://kr.api.riotgames.com/lol/league/v4/entries/by-summoner/{}?api_key={}"
    tempo_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
    temp_df = pd.DataFrame()
    api_urls = list(map(lambda sum: path.format(sum,main_api_key ), merged_large["{}_sumID".format(lane_team)]  ))
    if len(api_urls) != len(merged_large):
        print("creted api urls do not have the same length as the df, aborting and causing an error on purpose")
        #5 / 0
    for i in tqdm(range(len(api_urls))):
        try:
            api_url = api_urls[i]
            r = requests.get(api_url)
            trying = True
            count = 0
            count2  = 0
            while r.status_code == 429 or r.status_code == 504:

                if r.status_code == 504:
                    print("gateaway timeout")
                    count = count + 1
                    if count == 150:
                        break
                if trying:
                    tempo_api_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] , api_machine.switch())
                    tempo_r = requests.get(tempo_api_url)
                    if tempo_r.status_code == 429 or tempo_r.status_code == 504:
                        if tempo_r.status_code == 504:
                            count2 = count2  + 1
                            if count2 == 150:
                                break
                        continue
                    if tempo_r.status_code == 404:
                        trying = False
                        continue
                    try:
                        tempo_id = tempo_r.json()["id"]
                        api_url = path.format(tempo_id, api_machine.current_api_key)
                        r = requests.get(api_url)
                    except Exception as e:
                        print("some error {}".format(e))
                        trying = False
                        continue

                else:
                    try:
                        time.sleep(1)
                        api_url = api_url = api_urls[i]
                        r = requests.get(api_url)
                    except Exception as e:
                        pirnt("some kind of error {}".format(e))
                        break
            temp_df[i] =  pd.Series(r.json()[0] )
        except Exception as e:
            print("an error {}".format(e))
            print(i)
            temp_df[i] = None
    temp_df = temp_df.T.loc[:,["tier","rank","wins","losses","veteran","inactive","freshBlood","hotStreak"]]
    col_dic = {}
    for col in temp_df.columns.tolist():
        col_dic[col] = "{}_{}".format(lane_team , col)
    temp_df = temp_df.rename(columns = col_dic)
    if len(temp_df) == len(merged_large):
        merged_large = pd.concat( (merged_large, temp_df), axis = 1 )
    else:
        print("there is some thign wrong with the length")

    return merged_large


def get_win_los_rate_info_all_lanes(df, main_api_key, api_key_list,lanes):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    for lane in all_lanes:
        print(lane)
        merged_added = get_win_los_rate_info(merged_added, main_api_key , api_key_list,lane)
    return merged_added




def get_top10avg_champ_detail_info(df,main_api_key,  api_key_listing, lane_team):
    merged_large = df.copy()
    api_key_list = api_key_listing.copy()
    api_machine  = api_box(api_key_list)
    path = 'https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{}?api_key={}'
    tempo_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
    merged_large["{}_champion_levels".format(lane_team)] = None
    merged_large["{}_champion_points".format(lane_team)]  = None
    merged_large["{}_tokens".format(lane_team)] = None
    merged_large["{}_lastplaytime".format(lane_team)] = None
    merged_large["{}_ranking_favorite_list".format(lane_team)] = None


    merged_large["{}_avg_champion_levels".format(lane_team)] = None
    merged_large["{}_avg_champion_points".format(lane_team)] = None
    merged_large["{}_avg_tokens".format(lane_team)] = None
    merged_large["{}_avg_lastplaytime".format(lane_team)] = None
    picked_champion_ids = merged_large["{}_champ".format(lane_team)].tolist()
    api_urls_list = list(map(lambda sum: path.format(sum,main_api_key) , merged_large["{}_sumID".format(lane_team)]   ))

    if len(api_urls_list) != len(merged_large) or len(picked_champion_ids) != len(merged_large):
        print("created api urls do not have the same length as the df, aborting and causing an error on purpose")
        #5 / 0

    for i in tqdm(range(len(api_urls_list))):
        try:

            api_url = api_urls_list[i]
            r= requests.get(api_url)
            trying = True
            count = 0
            count2 = 0
            while r.status_code == 429 or r.status_code == 504 :
                # try:
                if r.status_code == 504:
                    count = count + 1
                    print("gateawaytimeout")
                    if count == 150:
                        break

                if trying:
                    tempo_api_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] , api_machine.switch() )
                    tempo_r = requests.get(tempo_api_url)

                    if tempo_r.status_code == 429 or tempo_r.status_code == 504:
                        if tempo_r.status_code == 504:
                            count2 = count2 + 1
                            if count2 == 150:
                                break

                        continue

                    if tempo_r.status_code == 404:
                        trying = False
                        continue
                    try:
                        tempo_id = tempo_r.json()["id"]
                        api_url = path.format(tempo_id, api_machine.current_api_key)
                        r  = requests.get(api_url)
                    except Exception as e:
                        print("something wrong : {}".format(e))
                        print(tempo_r.json())
                        trying = False
                        continue

                else:
                    time.sleep(2)
                    api_url = api_urls_list[i]
                    r = requests.get(api_url)



            all_json = r.json()
            current_json = all_json[:5] ###actually top 5
            picked_id =  picked_champion_ids[i]
            championId_list = list(map(lambda x: x["championId"] , all_json))
            if picked_id in championId_list:
                picked_index  =  championId_list.index(picked_id)
                picked_champion_info = all_json[picked_index]
                # print(picked_champion_info)
                merged_large["{}_champion_levels".format(lane_team)].iloc[i] = picked_champion_info["championLevel"]
                merged_large["{}_champion_points".format(lane_team)].iloc[i] = picked_champion_info["championPoints"]
                merged_large["{}_tokens".format(lane_team)].iloc[i] = picked_champion_info["tokensEarned"]
                merged_large["{}_lastplaytime".format(lane_team)].iloc[i] = picked_champion_info["lastPlayTime"]
                merged_large["{}_ranking_favorite_list".format(lane_team)].iloc[i] =  picked_index
            else:
                picked_champion_info = all_json[-1]
                #print(picked_champion_info)
                merged_large["{}_champion_levels".format(lane_team)].iloc[i] = picked_champion_info["championLevel"]
                merged_large["{}_champion_points".format(lane_team)].iloc[i] = picked_champion_info["championPoints"]
                merged_large["{}_tokens".format(lane_team)].iloc[i] = picked_champion_info["tokensEarned"]
                merged_large["{}_lastplaytime".format(lane_team)].iloc[i] = picked_champion_info["lastPlayTime"]
                merged_large["{}_ranking_favorite_list".format(lane_team)].iloc[i] =  picked_index
                merged_large["{}_ranking_favorite_list".format(lane_team)].iloc[i] = len(all_json)

            tempo_avg_champion_level =  mean(list(map(lambda x: int(x["championLevel"]), current_json  )))
            tempo_avg_champion_points = mean(list(map(lambda x: int(x["championPoints"]), current_json    )))
            tempo_avg_tokens = mean(list(map(lambda x:int(x["tokensEarned"]), current_json  )))
            tempo_avg_lastPlaytime = mean(list(map(lambda x : int(x["lastPlayTime"]),current_json )))
            merged_large["{}_avg_champion_levels".format(lane_team)].iloc[i] = tempo_avg_champion_level
            merged_large["{}_avg_champion_points".format(lane_team)].iloc[i] = tempo_avg_champion_points
            merged_large["{}_avg_tokens".format(lane_team)].iloc[i] = tempo_avg_tokens
            merged_large["{}_avg_lastplaytime".format(lane_team)].iloc[i] = tempo_avg_lastPlaytime
        except Exception as e:
            print("error : {} at {}".format(e ,i) )
            continue
    return merged_large

def champion_avg_detail_for_all_lanes(df,main_api_key,  api_key_list,lanes):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    for lane in all_lanes:
        print(lane)
        merged_added = get_top10avg_champ_detail_info(merged_added, main_api_key,  api_key_list,lane)
    return merged_added



def very_detail_champ_info_and_history(df , main_api_key, api_key_listing, lane_team):
    merged_large = df.copy()
    api_key_list = api_key_listing.copy()
    api_machine = api_box(api_key_list)
    all_apis = api_key_list.copy()
    all_apis.append(main_api_key)
    all_api_machine = api_box(all_apis)
    path = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/{}?api_key={}&champion={}&endIndex=3"
    tempo_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
    new_path = "https://kr.api.riotgames.com/lol/match/v4/matches/{}?api_key={}"

    merged_large["{}_total_champion_games".format(lane_team)] = None
    merged_large["{}_win_or_loss".format(lane_team)] = None
    merged_large["{}_kills".format(lane_team)] = None
    merged_large["{}_deats".format(lane_team)] = None
    merged_large["{}_assists".format(lane_team)] = None
    merged_large["{}_total_damage".format(lane_team)] = None
    merged_large["{}_damage_to_champ".format(lane_team)] = None
    merged_large["{}_damage_to_objects".format(lane_team)] = None
    merged_large["{}_champ_level".format(lane_team)] = None
    merged_large["{}_gold_earned".format(lane_team)] = None
    merged_large["{}_vision_score".format(lane_team)] = None
    merged_large["{}_minionskilled".format(lane_team)] = None
    merged_large["{}_neutral_minions_killed".format(lane_team)] = None
    merged_large["{}_wards_placed".format(lane_team)] = None
    merged_large["{}_wards_killed".format(lane_team)] = None
    merged_large["{}_damage_taken".format(lane_team)] = None



    all_accountId_list = merged_large["{}_accountId".format(lane_team)].tolist()
    all_champs_list = merged_large["{}_champ".format(lane_team)].tolist()
    api_url_list = list(map(lambda accountId, champion: path.format(accountId,main_api_key,champion),all_accountId_list,all_champs_list))
    if len(api_url_list) != len(merged_large):
        print("lengths are not the same aborting")
        #5/ 0
    for i in tqdm(range(len(api_url_list))):
        try:

            #time.sleep(1)
            current_championId = all_champs_list[i]
            api_url = api_url_list[i]
            r= requests.get(api_url)
            trying = True
            count = 0
            count2 = 0
            while r.status_code == 429 or r.status_code == 504:

                if r.status_code == 504:
                    time.sleep(1)
                    print("gateawaytimeout")
                    count  = count + 1
                    if count == 150:
                        break

                if trying:
                    tempo_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] , api_machine.switch() )
                    try:
                        tempo_r = requests.get(tempo_url)
                        if tempo_r.status_code == 429 or tempo_r.status_code ==504:
                            if temp_r.status_code == 504:
                                time.sleep(1)
                                print("gateway timeout in tempo")
                                count2 = count2 + 1
                                if count2 == 150:
                                    break
                            continue

                    except Exception as e:
                        print("error 1 {}".format(e))
                        print("tempo_r : {}".format(tempo_r.status_code))
                        trying = False
                        continue
                    if tempo_r.status_code == 404:
                        trying = False
                        print("tempo_r -> 404")
                        continue
                    try:
                        tempo_account = tempo_r.json()["accountId"]
                        api_url  = path.format(tempo_account , api_machine.current_api_key,  current_championId )
                        r = requests.get(api_url)
                    except Exception as e:
                        print("error soemthing {}".format(e))
                        print(tempo_r.json())
                        trying  = False
                        continue
                else:
                    time.sleep(3)
                    api_url = api_url_list[i]
                    r = requests.get(api_url)


            if  r.status_code == 404:
                merged_large["{}_total_champion_games".format(lane_team)].iloc[i] = 0
                continue

                #time.sleep(1)


            current_dict = r.json()
            merged_large["{}_total_champion_games".format(lane_team)].iloc[i] = current_dict["totalGames"]

            tempo_gameids = list(map(lambda x: x["gameId"], current_dict["matches"] ))


            win_or_loss = []
            kills = []
            deaths = []
            assists = []
            total_damage = []
            damage_to_champ = []
            damage_to_objects = []
            champ_level = []
            gold_earned = []
            vision_score = []
            minionskilled = []
            neutral_minions_killed = []
            wards_placed = []
            wards_killed = []
            damage_taken = []

            for each_game in tempo_gameids:
                try:
                    #time.sleep(1)
                    new_api_url = new_path.format(each_game, all_api_machine.switch())
                    new_r = requests.get(new_api_url)
                    count = 0
                    while new_r.status_code == 429 or new_r.status_code == 504:
                        ##
                        if new_r.status_code == 504:
                            time.sleep(1)
                            print("gateawaytimeout")
                            count = count + 1
                            if count == 150:
                                break
                        ##
                        time.sleep(1)
                        try:
                            new_api_url = new_path.format(each_game , all_api_machine.switch())
                            new_r = requests.get(new_api_url)
                        except Exception as e:
                            print("error 2 :{}".formate(e))

                        #time.sleep(1)
                    new_current_data = new_r.json()
                    game_duration = float(new_current_data["gameDuration"] / 100)
                    game_info =  new_current_data["participants"]
                    participant_index = list(map(lambda x: x["championId"] , game_info )).index(current_championId)

                    if "win" in game_info[participant_index]["stats"].keys():
                        win_or_loss.append(float(game_info[participant_index]["stats"]["win"]))
                    else:
                        print("no kills")

                    if "kills" in game_info[participant_index]["stats"].keys():
                        kills.append(float(game_info[participant_index]["stats"]["kills"]/ game_duration))
                    else:
                        print("no kills")

                    if "deaths" in game_info[participant_index]["stats"].keys():
                        deaths.append(float(game_info[participant_index]["stats"]["deaths"] / game_duration))
                    else:
                        print("no deaths")

                    if "assists" in game_info[participant_index]["stats"].keys():
                        assists.append(float(game_info[participant_index]["stats"]["assists"] / game_duration))
                    else:
                        print("no assists")

                    if "totalDamageDealt" in game_info[participant_index]["stats"].keys():
                        total_damage.append(float(game_info[participant_index]["stats"]["totalDamageDealt"] / game_duration))
                    else:
                        print("no totalDamageDealt")

                    if "totalDamageDealtToChampions" in game_info[participant_index]["stats"].keys():
                        damage_to_champ.append(float(game_info[participant_index]["stats"]["totalDamageDealtToChampions"] / game_duration))
                    else:
                        print("no totalDamageDealtToChampions")

                    if "damageDealtToObjectives" in game_info[participant_index]["stats"].keys():
                        damage_to_objects.append(float(game_info[participant_index]["stats"]["damageDealtToObjectives"] / game_duration))
                    else:
                        print("no damageDealtToObjectives")

                    if "champLevel" in game_info[participant_index]["stats"].keys():
                        champ_level.append(float(game_info[participant_index]["stats"]["champLevel"] / game_duration))
                    else:
                        print("no champLevel")

                    if "goldEarned" in game_info[participant_index]["stats"].keys():
                        gold_earned.append(float(game_info[participant_index]["stats"]["goldEarned"] / game_duration))
                    else:
                        print("no goldEarned")

                    if "visionScore" in game_info[participant_index]["stats"].keys():
                        vision_score.append(float(game_info[participant_index]["stats"]["visionScore"]))
                    else:
                        print("no visionScore")

                    if "totalMinionsKilled" in game_info[participant_index]["stats"].keys():
                        minionskilled.append(float(game_info[participant_index]["stats"]["totalMinionsKilled"] / game_duration))
                    else:
                        print(" no totalMinionsKilled")

                    if "neutralMinionsKilled"  in game_info[participant_index]["stats"].keys():
                        neutral_minions_killed.append(float(game_info[participant_index]["stats"]["neutralMinionsKilled"] / game_duration))
                    else:
                        print("no neutralMinionsKilled")

                    if "wardsPlaced" in game_info[participant_index]["stats"].keys():
                        wards_placed.append(float(game_info[participant_index]["stats"]["wardsPlaced"] / game_duration))
                    else:
                        print("no wards placed")

                    if "wardsKilled" in game_info[participant_index]["stats"].keys():
                        wards_killed.append(float(game_info[participant_index]["stats"]["wardsKilled"] / game_duration))
                    else:
                        print("no wards killed")

                    if "totalDamageTaken" in  game_info[participant_index]["stats"].keys():
                        damage_taken.append(float(game_info[participant_index]["stats"]["totalDamageTaken"] / game_duration))
                    else:
                        print("no totalDamageTaken")

                except Exception as e:
                    print("error 3 : {}".format(e))
                    print(new_r.status_code)
                    print(new_current_data)
                    continue

            if len(win_or_loss) >= 1:
                avg_win_or_loss = sum(win_or_loss) / len(win_or_loss)
            else:
                avg_win_or_loss = np.nan

            if len(kills) >= 1:
                avg_kills =  sum(kills) / len(kills)
            else:
                avg_kills = np.nan

            if len(deaths) >= 1:
                avg_deaths = sum(deaths) / len(deaths)
            else:
                avg_deaths = np.nan

            if len(assists) >= 1:
                avg_assists = sum(assists) / len(assists)
            else:
                avg_assists = np.nan

            if len(total_damage) >= 1:
                avg_total_damage = sum(total_damage) / len(total_damage)
            else:
                avg_total_damage = np.nan

            if len(damage_to_champ) >= 1:
                avg_damage_to_champ = sum(damage_to_champ) / len(damage_to_champ)
            else:
                avg_damage_to_champ = np.nan

            if len(damage_to_objects) >= 1:
                avg_damage_to_objects = sum(damage_to_objects) / len(damage_to_objects)
            else:
                avg_damage_to_objects = np.nan

            if len(champ_level) >= 1:
                avg_champ_level = sum(champ_level) / len(champ_level)
            else:
                avg_champ_level = np.nan

            if len(gold_earned) >= 1:
                avg_gold_earned = sum(gold_earned) / len(gold_earned)
            else:
                avg_gold_earned = np.nan

            if len(vision_score) >= 1:
                avg_vision_score = sum(vision_score) / len(vision_score)
            else:
                avg_vision_score = np.nan

            if len(minionskilled) >= 1:
                avg_minionskilled = sum(minionskilled) / len(minionskilled)
            else:
                avg_minionskilled = np.nan

            if len(neutral_minions_killed) >= 1:
                avg_neutral_minions_killed = sum(neutral_minions_killed) / len(neutral_minions_killed)
            else:
                avg_neutral_minions_killed = np.nan

            if len(wards_placed) >= 1:
                avg_wards_placed = sum(wards_placed) / len(wards_placed)
            else:
                avg_wards_placed = np.nan

            if len(wards_killed) >= 1:
                avg_wards_killed = sum(wards_killed) / len(wards_killed)
            else:
                avg_wards_killed = np.nan

            if len(damage_taken) >= 1:
                avg_damage_taken = sum(damage_taken) / len(damage_taken)
            else:
                avg_damage_taken = np.nan

            merged_large["{}_win_or_loss".format(lane_team)].iloc[i] = avg_win_or_loss
            merged_large["{}_kills".format(lane_team)].iloc[i] = avg_kills
            merged_large["{}_deats".format(lane_team)].iloc[i] = avg_deaths
            merged_large["{}_assists".format(lane_team)].iloc[i] = avg_assists
            merged_large["{}_total_damage".format(lane_team)].iloc[i] = avg_total_damage
            merged_large["{}_damage_to_champ".format(lane_team)].iloc[i] = avg_damage_to_champ
            merged_large["{}_damage_to_objects".format(lane_team)].iloc[i] = avg_damage_to_objects
            merged_large["{}_champ_level".format(lane_team)].iloc[i] = avg_champ_level
            merged_large["{}_gold_earned".format(lane_team)].iloc[i] = avg_gold_earned
            merged_large["{}_vision_score".format(lane_team)].iloc[i] = avg_vision_score
            merged_large["{}_minionskilled".format(lane_team)].iloc[i] = avg_minionskilled
            merged_large["{}_neutral_minions_killed".format(lane_team)].iloc[i] = avg_neutral_minions_killed
            merged_large["{}_wards_placed".format(lane_team)].iloc[i] = avg_wards_placed
            merged_large["{}_wards_killed".format(lane_team)].iloc[i] = avg_wards_killed
            merged_large["{}_damage_taken".format(lane_team)].iloc[i] = avg_damage_taken

        except Exception as e:
            print("error total : {}".format(e))
            continue
    return merged_large



def very_detail_champ_info_and_history_for_all_lanes(df , main_api_key, api_key_listing, lane_team):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    api_key_list = api_key_listing.copy()
    for lane in all_lanes:
        print(lane)
        merged_added = very_detail_champ_info_and_history(merged_added , main_api_key, api_key_list, lane)
    return merged_added




def coerce_df_columns_to_numeric(df):
    columns = []
    for column in df.columns.tolist():
        if is_numeric_dtype(df[column]):
            columns.append(column)
    column_list = columns
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
    return df

def del_nan_merge_large(df):
    merged_large = df.copy()
    merged_large = merged_large.dropna()
    merged_large = coerce_df_columns_to_numeric(merged_large)
    return merged_large



def final_final_modify(final_data):

    del final_data["participants"]
    del final_data["participantIdentities"]
    del final_data["teams"]

    for i in final_data.columns:
        try:
            final_data[i] = pd.to_numeric(final_data[i], errors = 'ignore')
        except:
            continue
    return final_data




def change_champID_to_champName(df ,general_champ_df ,lanes):
    champ_info = general_champ_df.copy()
    all_lanes = lanes
    merged_added = df.copy()
    merged_added2 = pd.DataFrame()
    for lane in tqdm(lanes):
        col_name = "{}_champ".format(lane)
        str_id = list(map(lambda x: str(x)    , merged_added[col_name] ))
        champion_names = champion_key_to_id(champ_info,  str_id ).tolist()
        merged_added2["{}_champName".format(lane)] = champion_names
    return merged_added2

# testing = change_champID_to_champName(merged_added, champ_info , all_lanes)






# import datetime
# main_api_key = api_config.main_api_key
#
# api_key1 = "RGAPI-97f5f62b-8ccb-4b9d-b34c-554b9f4b4499"
# api_key2 = "RGAPI-15aefc8d-50cd-4a90-81fd-af88acd38312"
# api_key3 = "RGAPI-4d18bd3f-e718-4cb3-8ce6-c32d59d0987c"
# api_key_list = [api_key1 , api_key2, api_key3]
#
#
#
#
# a = datetime.datetime.now()
# gm_df = show_grandmaster_info(main_api_key)
# gm_df = df_summoner_accountid(gm_df, main_api_key)
# match_info_df =  accountID_to_matchINFO(league_df3 = gm_df, endIndex=2, api_key= main_api_key)
# match_info_df =  match_info_df.drop_duplicates(subset = "gameId").reset_index(drop = True)
# match_df = game_id_to_match_detail(match_info_df, main_api_key)
# match_df  = modifiy_match_df_original(match_df)
# b = datetime.datetime.now()
# match_df.to_csv("match_df.csv")
# print(b - a )
#
#
# c = datetime.datetime.now()
# match_df = match_df.drop_duplicates(subset = "gameId").reset_index(drop=True)
# match_time_list = get_time_line_list(match_df, main_api_key, api_key_list)
# spell = spell_general_info()
# lane_matching_df = participants_for_lanes(match_df, match_time_list)
# lane_info = modify_lane_matching_df(lane_matching_df)
# merged_info = merge_lane_info_to_match_info(match_df, lane_info)
# merged_info = modify_merged_info(merged_info)
# merged_info.to_csv("middle_point_saving1.csv")
# d = datetime.datetime.now()
# print(d - c)
#
#
#
#
#
# e = datetime.datetime.now()
# merged_info = get_win_loss_col(merged_info)
# merged_added = get_champion_sumId_cols(merged_info)
# all_lanes =["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200", "SUPPORT200"]
# merged_added = get_summonerLevel_for_all_lanes(merged_added,main_api_key, api_key_list,all_lanes)
# merged_added = merged_added.dropna()
# merged_added = get_win_los_rate_info_all_lanes(merged_added , main_api_key , api_key_list ,all_lanes )
# merged_added = merged_added.dropna()
# merged_added.to_csv("middle_point_saving2.csv")
# f = datetime.datetime.now()
# print(f - e)
#
#
#
# g = datetime.datetime.now()
# merged_checking= champion_avg_detail_for_all_lanes(merged_added , main_api_key , api_key_list , all_lanes)
# h = datetime.datetime.now()
# print(h-g)
# merged_checking.to_csv("middle_point_saving3.csv")
#
#
# ii = datetime.datetime.now()
# merged_checking = merged_checking.dropna()  ########### dropping na
# merged_final = very_detail_champ_info_and_history_for_all_lanes(merged_checking, main_api_key, api_key_list, all_lanes )
# final_final = final_final_modify(merged_final)
# j = datetime.datetime.now()
# print(j-ii)
# final_final.to_csv("final_final.csv")
#
#
# pymysql.install_as_MySQLdb()
# host = "192.168.0.181"
# db_name = "lolpred"
# user_name = "root"
# password = "123"
# port = 3306
# db_type = "mysql"
# connect_db = connect_sql(host,db_name,user_name,password,port, db_type)
# connect_db.insert_df(final_final , "grandmaster_0805)


main_api_key = api_config.main_api_key
api_key2 = "RGAPI-65454244-792a-4941-a11e-629154eeff9c"
api_key3 = 'RGAPI-ad0fb4d1-861b-4b6c-8225-24fd12996850'
api_key1 = 'RGAPI-10b7e0b9-7424-4db6-9a18-d4b55eff2f2c'




if __name__ == '__main__':
    import datetime
    main_api_key = api_config.main_api_key

    api_key1 = "RGAPI-97f5f62b-8ccb-4b9d-b34c-554b9f4b4499"
    api_key2 = "RGAPI-15aefc8d-50cd-4a90-81fd-af88acd38312"
    api_key3 = "RGAPI-4d18bd3f-e718-4cb3-8ce6-c32d59d0987c"
    api_key_list = [api_key1 , api_key2, api_key3]
    all_lanes =["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200", "SUPPORT200"]

    a = datetime.datetime.now()
    gm_df = show_grandmaster_info(main_api_key)
    gm_df = df_summoner_accountid(gm_df, main_api_key)
    match_info_df =  accountID_to_matchINFO(league_df3 = gm_df, endIndex=2, api_key= main_api_key)
    match_info_df =  match_info_df.drop_duplicates(subset = "gameId").reset_index(drop = True)
    match_df = game_id_to_match_detail(match_info_df, main_api_key)
    match_df  = modifiy_match_df_original(match_df)
    b = datetime.datetime.now()
    match_df.to_csv("match_df.csv")
    print(b - a )


    c = datetime.datetime.now()
    match_df = match_df.drop_duplicates(subset = "gameId").reset_index(drop=True)
    match_time_list = get_time_line_list(match_df, main_api_key, api_key_list)
    spell = spell_general_info()
    lane_matching_df = participants_for_lanes(match_df, match_time_list)
    lane_info = modify_lane_matching_df(lane_matching_df)
    merged_info = merge_lane_info_to_match_info(match_df, lane_info)
    merged_info = modify_merged_info(merged_info)
    merged_info.to_csv("middle_point_saving1.csv")
    d = datetime.datetime.now()
    print(d - c)

    ##
    merged_info = pd.read_csv("middle_point_saving1.csv" , index_col = 0)
    for column in ['teams', 'participants', 'participantIdentities']:
        merged_info[column] = merged_info[column].map(lambda v: eval(v))

    ###
    e = datetime.datetime.now()
    merged_info = get_win_loss_col(merged_info)
    merged_added = get_champion_sumId_cols(merged_info)
    merged_added = get_summonerLevel_for_all_lanes(merged_added,main_api_key, api_key_list,all_lanes)
    merged_added = merged_added.dropna()
    merged_added = get_win_los_rate_info_all_lanes(merged_added , main_api_key , api_key_list ,all_lanes )
    merged_added = merged_added.dropna()
    merged_added.to_csv("middle_point_saving2.csv")
    f = datetime.datetime.now()
    print(f - e)


    g = datetime.datetime.now()
    merged_checking= champion_avg_detail_for_all_lanes(merged_added , main_api_key , api_key_list , all_lanes)
    h = datetime.datetime.now()
    print(h-g)
    merged_checking.to_csv("middle_point_saving3.csv")


    ii = datetime.datetime.now()
    merged_checking = merged_checking.dropna()  ########### dropping na
    merged_final = very_detail_champ_info_and_history_for_all_lanes(merged_checking, main_api_key, api_key_list, all_lanes )
    final_final = final_final_modify(merged_final)
    j = datetime.datetime.now()
    print(j-ii)
    final_final.to_csv("final_final.csv")


    # pymysql.install_as_MySQLdb()
    # host = "192.168.0.181"
    # db_name = "lolpred"
    # user_name = "root"
    # password = "123"
    # port = 3306
    # db_type = "mysql"
    # connect_db = connect_sql(host,db_name,user_name,password,port, db_type)
    # connect_db.insert_df(final_final , "grandmaster_0805")


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
#
# data1, data2 = testing(df, main_api_key, api_key_list, lane_team)
#
#
# def testing(df, main_api_key, api_key_list, lane_team):
#     merged_large = df.copy()
#     api_machine  = api_box(api_key_list)
#     path = "https://kr.api.riotgames.com/lol/league/v4/entries/by-summoner/{}?api_key={}"
#     tempo_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
#     temp_df = pd.DataFrame()
#     api_urls = list(map(lambda sum: path.format(sum,main_api_key ), merged_large["{}_sumID".format(lane_team)]  ))
#     if len(api_urls) != len(merged_large):
#         print("creted api urls do not have the same length as the df, aborting and causing an error on purpose")
#         #5 / 0
#     finish = False
#     checking_index = 0
#     for i in tqdm(range(len(api_urls))):
#         try:
#             api_url = api_urls[i]
#             r = requests.get(api_url)
#             trying = True
#             count = 0
#             count2  = 0
#             if r.status_code == 429:
#                 print("now it is going through different api")
#             while r.status_code == 429 or r.status_code == 504:
#
#                 if r.status_code == 504:
#                     print("gateaway timeout")
#                     count = count + 1
#                     if count == 150:
#                         break
#                 if trying:
#                     tempo_api_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] , api_machine.switch())
#                     tempo_r = requests.get(tempo_api_url)
#                     if tempo_r.status_code == 200:
#                         print("first step worked properly")
#                     if tempo_r.status_code == 429 or tempo_r.status_code == 504:
#                         if tempo_r.status_code == 504:
#                             count2 = count2  + 1
#                             if count2 == 150:
#                                 break
#                         continue
#                     if tempo_r.status_code == 404:
#                         trying = False
#                         continue
#                     try:
#                         tempo_id = tempo_r.json()["id"]
#                         api_url = path.format(tempo_id, api_machine.current_api_key)
#                         r = requests.get(api_url)
#                         if r.status_code == 200:
#                             print("second step worked properly")
#                             checking_index = i
#                             finish = True
#                             data1 = r.json()
#                             break
#                     except Exception as e:
#                         print("some error {}".format(e))
#                         trying = False
#                         continue
#
#                 else:
#                     try:
#                         time.sleep(1)
#                         api_url = api_url = api_urls[i]
#                         r = requests.get(api_url)
#                     except Exception as e:
#                         pirnt("some kind of error {}".format(e))
#                         break
#             if finish:
#                 break
#
#         except Exception as e:
#             print("an error {}".format(e))
#             print(i)
#     time.sleep(30)
#     print("checking_index {}".format(checking_index))
#     testing_url =api_urls[checking_index]
#     testing_r = requests.get(testing_url)
#     while testing_r.status_code == 429:
#         time.sleep(1)
#         testing_r = requests.get(testing_url)
#     data2 = testing_r.json()
#     return data1 , data2
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
