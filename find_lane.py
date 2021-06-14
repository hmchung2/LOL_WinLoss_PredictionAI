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

from skimage import io # 미니맵 처리
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_numeric_dtype
from statistics import mean
import api_config
import api_logging
from sqlalchemy import create_engine, types, select
from sqlalchemy import *

from packaging import version
import datetime


# MySQL Connector using pymysql
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
def df_summoner_accountid(league_df,api_key , log , error_log):
    league_df['account_id'] = None
    for i in range(len(league_df)):
        try:
            #sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key
            sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/' + league_df['summonerId'].iloc[i] + '?api_key=' + api_key
            r = requests.get(sohwan)

            while r.status_code == 429 or r.status_code == 504:
                time.sleep(3)
                print('time to wait')
                r = requests.get(sohwan)
            account_id = r.json()['accountId']

            league_df.iloc[i, -1] = account_id
        except Exception as e:
            error_log.error('df_summoner_accountid error at iteration {} ---> {}'.format( i, e))
            pass
    return league_df

########################### not interchangeable ##########################################
def accountID_to_matchINFO(league_df3, endIndex , api_key , log , error_log):
    log.info("procssing ---> accountId_to_matchINFO")
    # need account_id column in the data frame
    match_info_df = pd.DataFrame()
    season = api_config.Config.season
    EI = str(endIndex)
    start =datetime.datetime.now()
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
        except Exception as e:
            error_log.error("accountId_to_matchINFO error at iteration : {} ---> {}".format(i , e))
            continue
            #print('not going good')
            #print(i)
    end = datetime.datetime.now()
    dur = end - start
    seconds = dur.seconds
    log.info("accountID_to_matchINFO process duration ---> : {} seconds".format(seconds))
    return match_info_df

# Match 데이터 받기 (gameId를 통해 경기의 승패, 팀원과 같은 정보가 담겨있다.)
################# interchangeable but not recommending since we will use the data in the futrue  #########################33
def game_id_to_match_detail(match_info_df2, api_key, log , error_log):
    log.info("processing ---> game_id_to_match_detail")
    start =datetime.datetime.now()
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
        except Exception as e:
            error_log.error("game_id_to_match_detail error at iteration : {} ---> {}".format(i , e))
            #print('not going well')
            pass
    match_fin2 = match_fin.reset_index()
    del match_fin2['index']
    end = datetime.datetime.now()
    dur = end - start
    seconds = dur.seconds
    log.info("game_id_to_match_detail process duration ---> {} seconds".format(seconds))
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
def modify_match_df_original(df):
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



def get_time_line_list(df,api_key , log , error_log):
    log.info("processing ---> get_time_line_list")
    start = datetime.datetime.now()
    match_df = df.copy()
    match_timeline_list = []
    for game_id in match_df['gameId']: # 각 게임 아이디마다 요청
        api_url = 'https://kr.api.riotgames.com/lol/match/v4/timelines/by-match/{}?api_key={}'.format(game_id , api_key)
        try:
            r = requests.get(api_url)
            while r.status_code == 429 or r.status_code == 504:
                time.sleep(3)
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
        except Exception as e:
            error_log.error("get_time_line_list error at game {} ---> {}".format(game_id , e))
            match_timeline_list.append(pd.DataFrame())
            continue
    end = datetime.datetime.now()
    dur = end - start
    seconds = dur.seconds
    log.info("get_time_line_list process duration ---> {} seconds".format(seconds))
    return match_timeline_list



def participants_for_lanes(match, timeline, log , error_log):
    log.info("processing ---> participants_for_lanes")
    start = datetime.datetime.now()
    match_df = match.copy()
    match_timeline_list = timeline.copy()
    lane_calculated = pd.DataFrame()
    for k in range(len(match_timeline_list)) :              #len(match_timeline_list)
        try:
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
        except Exception as e:
            error_log.error("participants_for_lanes error at iteration {} ---> {}".format(k , e))

    end = datetime.datetime.now()
    dur =  end - start
    seconds = dur.seconds
    log.info("participants_for_lanes process duration ---> {} seconds".format(seconds))

    return lane_calculated.T



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





def modify_lane_matching_df(df):
    lane_matching_df = df.copy()
    lane_matching_df = lane_matching_df.reset_index(drop = True)
    lane_matching_df = lane_matching_df.dropna()
    for col in lane_matching_df.columns.tolist():
        lane_matching_df[col] = list(map(lambda x: int(x)  , lane_matching_df[col] ))

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

def coerce_df_columns_to_numeric(df):
    columns = []
    for column in df.columns.tolist():
        if is_numeric_dtype(df[column]):
            columns.append(column)
    column_list = columns
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
    return df

def get_summonerLevel_for_all_lanes(df, api_key ,all_lanes , log , error_log):
    log.info("processing ---> get_summonerLevel_for_all_lanes")
    start = datetime.datetime.now()
    merged_added = df.copy()
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    for lane in all_lanes:
        log.info("get_summonerLevel_for_all_lanes at lane : {}".format(lane))
        merged_added = get_summonerLevel(merged_added, api_key,lane , log , error_log)
    end = datetime.datetime.now()
    dur =  end - start
    seconds = dur.seconds
    log.info("get_summonerLevel_for_all_lanes duration ----> {}".format(seconds))
    return merged_added



def get_summonerLevel(df, api_key,lane_team , log ,error_log):
    log.info("processing ---> get_summonerLevel")
    start =datetime.datetime.now()
    merged_large = df.copy()
    path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-account/{}?api_key={}"
    merged_large["{}_summonerLevel".format(lane_team)] = None
    for i in range(len(merged_large  )):
        api_url = path.format(merged_large["{}_accountId".format(lane_team)].iloc[i] , main_api_key )
        try:
            r = requests.get(api_url)
            while r.status_code == 429 or r.status_code ==504:
                time.sleep(3)
                r = requests.get(api_url)
            merged_large["{}_summonerLevel".format(lane_team)].iloc[i] = r.json()["summonerLevel"]
        except Exception as e:
            error_log.error("get_summonerLevel error at iteration {} ---> {}".format(i , e))
            error_log.error(api_url)
    end = datetime.datetime.now()
    dur = end - start
    seconds = dur.seconds
    log.info( "{} seconds".format(seconds))

    return merged_large


def get_win_los_rate_info(df, main_api_key, lane_team , log , error_log):
    merged_large = df.copy()
    path = "https://kr.api.riotgames.com/lol/league/v4/entries/by-summoner/{}?api_key={}"
    temp_df = pd.DataFrame()
    api_urls = list(map(lambda sum: path.format(sum,main_api_key ), merged_large["{}_sumID".format(lane_team)]  ))

    for i in range(len(api_urls)):
        try:
            api_url = api_urls[i]
            r = requests.get(api_url)
            while r.status_code == 429 or r.status_code == 504:
                time.sleep(2)
                r = requests.get(api_url)

            temp_df[i] =  pd.Series(r.json()[0] )
        except Exception as e:
            print("an error {}".format(e))
            temp_df[i] = None
    temp_df = temp_df.T.loc[:,["tier","rank","wins","losses","veteran","inactive","freshBlood","hotStreak"]]
    col_dic = {}
    for col in temp_df.columns.tolist():
        col_dic[col] = "{}_{}".format(lane_team , col)
    temp_df = temp_df.rename(columns = col_dic)
    #temp_df 와 merged_large 행 개수는 무조건 똑같아야한다.
    merged_large = pd.concat( (merged_large, temp_df), axis = 1 )
    return merged_large




def get_win_los_rate_info_all_lanes(df, main_api_key, lanes , log , error_log):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    for lane in lanes:
        log.info(" get_win_los_rate_info_all_lanes : {}".format(lane))
        merged_added = get_win_los_rate_info(merged_added, main_api_key ,lane , log , error_log)
    return merged_added




def get_top10avg_champ_detail_info(df,main_api_key, lane_team , log , error_log):
    merged_large = df.copy()
    path = 'https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{}?api_key={}'
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
    for i in range(len(api_urls_list)):
        try:
            api_url = api_urls_list[i]
            r= requests.get(api_url)

            while r.status_code == 429 or r.status_code == 504 :
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
                merged_large["{}_champion_levels".format(lane_team)].iloc[i] = picked_champion_info["championLevel"]
                merged_large["{}_champion_points".format(lane_team)].iloc[i] = picked_champion_info["championPoints"]
                merged_large["{}_tokens".format(lane_team)].iloc[i] = picked_champion_info["tokensEarned"]
                merged_large["{}_lastplaytime".format(lane_team)].iloc[i] = picked_champion_info["lastPlayTime"]
                merged_large["{}_ranking_favorite_list".format(lane_team)].iloc[i] =  picked_index
            else:
                picked_champion_info = all_json[-1]
                merged_large["{}_champion_levels".format(lane_team)].iloc[i] = picked_champion_info["championLevel"]
                merged_large["{}_champion_points".format(lane_team)].iloc[i] = picked_champion_info["championPoints"]
                merged_large["{}_tokens".format(lane_team)].iloc[i] = picked_champion_info["tokensEarned"]
                merged_large["{}_lastplaytime".format(lane_team)].iloc[i] = picked_champion_info["lastPlayTime"]
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
            error_log.error("get_top10avg_champ_detail_info : {} :{}".format(i , e))
            continue
    return merged_large




def champion_avg_detail_for_all_lanes(df,main_api_key,lanes , log, error_log):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    for lane in all_lanes:
        log.info("champion_avg_detail_for_all_lanes : {}".format(lane))
        merged_added = get_top10avg_champ_detail_info(merged_added, main_api_key,lane , log , error_log)
    return merged_added



def very_detail_champ_info_and_history(df , main_api_key, lane_team , log , error_log):
    merged_large = df.copy()
    path = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/{}?api_key={}&champion={}&endIndex=6"
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
    gameId_list = [x for x in merged_large["gameId"]]
    for i in range(len(api_url_list)):

        try:
            log.info("very_detail_champ_info_and_history account_index : {}".format(i))
            #time.sleep(1)
            current_championId = all_champs_list[i]
            api_url = api_url_list[i]
            current_gameId = gameId_list[i]
            r= requests.get(api_url)
            while r.status_code == 429 or r.status_code == 504:
                time.sleep(3)
                api_url = api_url_list[i]
                r = requests.get(api_url)
            if  r.status_code == 404:
                merged_large["{}_total_champion_games".format(lane_team)].iloc[i] = 0
                continue
            current_dict = r.json()

            merged_large["{}_total_champion_games".format(lane_team)].iloc[i] = current_dict["totalGames"]
            current_dict["matches"] = [current_dict["matches"][x] for x in range(len(current_dict["matches"])) if current_dict["matches"][x]["gameId"] != current_gameId]
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
                    new_api_url = new_path.format(each_game,main_api_key)


                    new_r = requests.get(new_api_url)
                    while new_r.status_code == 429 or new_r.status_code == 504:
                        time.sleep(1)
                        new_r = requests.get(new_api_url)


                    new_current_data = new_r.json()
                    game_duration = float(new_current_data["gameDuration"] / 100)
                    game_info =  new_current_data["participants"]
                    participant_index = list(map(lambda x: x["championId"] , game_info )).index(current_championId)

                    if "win" in game_info[participant_index]["stats"].keys():
                        win_or_loss.append(float(game_info[participant_index]["stats"]["win"]))
                    if "kills" in game_info[participant_index]["stats"].keys():
                        kills.append(float(game_info[participant_index]["stats"]["kills"]/ game_duration))
                    if "deaths" in game_info[participant_index]["stats"].keys():
                        deaths.append(float(game_info[participant_index]["stats"]["deaths"] / game_duration))
                    if "assists" in game_info[participant_index]["stats"].keys():
                        assists.append(float(game_info[participant_index]["stats"]["assists"] / game_duration))
                    if "totalDamageDealt" in game_info[participant_index]["stats"].keys():
                        total_damage.append(float(game_info[participant_index]["stats"]["totalDamageDealt"] / game_duration))
                    if "totalDamageDealtToChampions" in game_info[participant_index]["stats"].keys():
                        damage_to_champ.append(float(game_info[participant_index]["stats"]["totalDamageDealtToChampions"] / game_duration))
                    if "damageDealtToObjectives" in game_info[participant_index]["stats"].keys():
                        damage_to_objects.append(float(game_info[participant_index]["stats"]["damageDealtToObjectives"] / game_duration))
                    if "champLevel" in game_info[participant_index]["stats"].keys():
                        champ_level.append(float(game_info[participant_index]["stats"]["champLevel"] / game_duration))
                    if "goldEarned" in game_info[participant_index]["stats"].keys():
                        gold_earned.append(float(game_info[participant_index]["stats"]["goldEarned"] / game_duration))
                    if "visionScore" in game_info[participant_index]["stats"].keys():
                        vision_score.append(float(game_info[participant_index]["stats"]["visionScore"]))
                    if "totalMinionsKilled" in game_info[participant_index]["stats"].keys():
                        minionskilled.append(float(game_info[participant_index]["stats"]["totalMinionsKilled"] / game_duration))
                    if "neutralMinionsKilled"  in game_info[participant_index]["stats"].keys():
                        neutral_minions_killed.append(float(game_info[participant_index]["stats"]["neutralMinionsKilled"] / game_duration))
                    if "wardsPlaced" in game_info[participant_index]["stats"].keys():
                        wards_placed.append(float(game_info[participant_index]["stats"]["wardsPlaced"] / game_duration))
                    if "wardsKilled" in game_info[participant_index]["stats"].keys():
                        wards_killed.append(float(game_info[participant_index]["stats"]["wardsKilled"] / game_duration))
                    if "totalDamageTaken" in  game_info[participant_index]["stats"].keys():
                        damage_taken.append(float(game_info[participant_index]["stats"]["totalDamageTaken"] / game_duration))
                except Exception as e:
                    error_log.error("very_detail_champ_info_and_history each game : {}".format(e))
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


def very_detail_champ_info_and_history_for_all_lanes(df , main_api_key, all_lanes , log ,error_log):
    merged_added = df.copy()
    for lane in all_lanes:
        log.info("very_detail_champ_info_and_history : {}".format( lane ) )
        merged_added = very_detail_champ_info_and_history(merged_added , main_api_key, lane , log, error_log)
    return merged_added





def coerce_df_columns_to_numeric(df):
    columns = []
    for column in df.columns.tolist():
        if is_numeric_dtype(df[column]):
            columns.append(column)
    column_list = columns
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
    return df



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


if __name__ == '__main__':

    main_api_key = api_config.main_api_key

    all_lanes =["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200", "SUPPORT200"]

    log = api_logging.get_log_view(1, "platform", False, 'create_data_log')
    error_log = api_logging.get_log_view(1, "platform" , True, 'create_data_error_log')


    gm_df = show_grandmaster_info(main_api_key)

    gm_df = gm_df.iloc[:200 , : ]

    gm_df.to_csv("final_final1005.csv")

    gm_df = df_summoner_accountid(gm_df, main_api_key , log ,error_log)

    match_info_df =  accountID_to_matchINFO(league_df3 = gm_df, endIndex=2, api_key= main_api_key , log  = log ,error_log = error_log)

    match_info_df =  match_info_df.drop_duplicates(subset = "gameId").reset_index(drop = True)

    match_df = game_id_to_match_detail(match_info_df, main_api_key , log , error_log)

    match_df  = modify_match_df_original(match_df)

    match_df = match_df.drop_duplicates(subset = "gameId").reset_index(drop=True)

    match_time_list = get_time_line_list(match_df, main_api_key , log ,error_log)

    spell = spell_general_info()

    lane_matching_df = participants_for_lanes(match_df, match_time_list, log ,error_log)

    lane_info = modify_lane_matching_df(lane_matching_df)

    merged_info = merge_lane_info_to_match_info(match_df, lane_info)

    merged_info = modify_merged_info(merged_info)

    merged_info = get_win_loss_col(merged_info)

    merged_added = get_champion_sumId_cols(merged_info)

    merged_added = get_summonerLevel_for_all_lanes(merged_added,main_api_key,all_lanes , log, error_log)

    merged_added = merged_added.dropna()

    merged_added = get_win_los_rate_info_all_lanes(merged_added , main_api_key  ,all_lanes , log, error_log)

    merged_added = merged_added.dropna()

    merged_checking = champion_avg_detail_for_all_lanes(merged_added,main_api_key,all_lanes , log, error_log)

    merged_checking = merged_checking.dropna()  ########### dropping na

    merged_final = very_detail_champ_info_and_history_for_all_lanes(merged_checking , main_api_key, all_lanes , log ,error_log)


    final_final = final_final_modify(merged_final)
    from datetime import datetime
    now = datetime.now()
    data_path = "data_storage/final_data"+ str(now.strftime("%y%m%d%H%M%S")) + ".csv"

    final_final.to_csv(data_path)



# pymysql.install_as_MySQLdb()
# host = "192.168.0.181"
# db_name = "lolpred"
# user_name = "root"
# password = "123"
# port = 3306
# db_type = "mysql"
# connect_db = connect_sql(host,db_name,user_name,password,port, db_type)
# connect_db.insert_df(final_final , "grandmaster_0805")
