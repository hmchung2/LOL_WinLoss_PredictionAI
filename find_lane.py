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

main_api_key = api_config.main_api_key

%matplotlib inline
api_key1 = "RGAPI-688d1da6-a461-478b-a273-8e64ce863324"
api_key2 = "RGAPI-804ac2a3-7a07-47a9-a6dd-c098ea335c3a"

api_key_list = [api_key1 , api_key2]


def get_current_version(key):
    api_key = key
    r = requests.get('https://ddragon.leagueoflegends.com/api/versions.json') # version data 확인
    current_version = r.json()[0]
    return current_version

current_ver = get_current_version(api_key1)
current_ver
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


champ_info = get_champion_id_by_current_version(main_api_key ,current_ver)
champ_info


def champion_key_to_id(champ_info,key_list):
    new_df = champ_info.set_index(pd.Series(champ_info.key.tolist()))
    champion_names = new_df.loc[key_list,'id']
    return champion_names

champion_key_to_id(champ_info,["26","142"])



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

show_grandmaster_info(main_api_key)

#leagueId = show_info(api_key,'GOLD',1, 2)["leagueId"][0]

#######################3 not interchangeable #############################
def df_summoner_accountid(league_df,api_key):
    league_df['account_id'] = None
    for i in range(len(league_df)):
        try:
            sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key
            r = requests.get(sohwan)

            while r.status_code == 429:
                time.sleep(5)
                print('time to wait')
                sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key
                r = requests.get(sohwan)

            account_id = r.json()['accountId']

            league_df.iloc[i, -1] = account_id
            print('going good')
        except:
            print('not ok')
            pass
    print('done')
    return league_df

#
gm_df = show_grandmaster_info(main_api_key  )
gm_df = gm_df.loc[:5,:]
#

gm_df = df_summoner_accountid(gm_df, main_api_key )

league_df = gm_df.loc[:2,:]

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
                print('time to wait')
                match0 = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + league_df3['account_id'].iloc[i]  +'?season=' + season +'&endIndex='+ EI+'&api_key=' + api_key
                r = requests.get(match0)

            match_info_df = pd.concat([match_info_df, pd.DataFrame(r.json()['matches'])])
            print('going good')
        except:
            print('not going good')
            print(i)
    print("done")
    return match_info_df


match_info_df =  accountID_to_matchINFO(league_df3 = league_df, endIndex=5, api_key= main_api_key)
match_info_df = match_info_df.reset_index(drop =True )
match_info_df
# Match 데이터 받기 (gameId를 통해 경기의 승패, 팀원과 같은 정보가 담겨있다.)
################# interchangeable but not recommending since we will use the data in the futrue  #########################33
def game_id_to_match_detail(match_info_df2, api_key):
    match_fin = pd.DataFrame()
    for i in range(len(match_info_df2)):
        try:
            api_url='https://kr.api.riotgames.com/lol/match/v4/matches/' + str(match_info_df2['gameId'].iloc[i]) + '?api_key=' + api_key
            r = requests.get(api_url)

            while r.status_code == 429:
                time.sleep(2)
                print('time to wait')
                #time.sleep를 꼭 해줘야함 안그러면 request 잦은 사용으로 블랙리스트가 됨
                api_url='https://kr.api.riotgames.com/lol/match/v4/matches/' + str(match_info_df2['gameId'].iloc[i]) + '?api_key=' + api_key
                r = requests.get(api_url)

            mat = pd.DataFrame(list(r.json().values()), index=list(r.json().keys())).T
            match_fin = pd.concat([match_fin,mat])
            print('going well')
        except:
            print('not going well')
            pass
    match_fin2 = match_fin.reset_index()
    del match_fin2['index']
    print('done')
    return match_fin2



match_df = game_id_to_match_detail(match_info_df, main_api_key)
match_df.to_csv('MatchData.csv') # 파일로 저장
match_info_df2 = merged_large.copy()
api_url='https://kr.api.riotgames.com/lol/match/v4/matches/' + str(match_info_df2['gameId'].iloc[i]) + '?api_key=' + main_api_key
r = requests.get(api_url)
r.json()

######################################################



match_df2 = pd.read_csv('MatchData.csv', index_col=0)



# when running in module withou stopping
def modifiy_match_df_original(df):
    match_df = df.copy()
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

def modfify_match_df(df):
    match_df = df.copy()
    match_df["gameId"] = list(map(lambda x: int(x), match_df["gameId"]))
    match_df["queueId"] = match_df["queueId"].map(lambda x : int(x))
    match_df["gameCreation"] = match_df["gameCreation"].map(lambda x : int(x))
    match_df["seasonId"] = match_df["seasonId"].map(lambda x : int(x))
    match_df["mapId"] = match_df["mapId"].map(lambda x : int(x))
    match_df["gameDuration"] = match_df["gameDuration"].map(lambda x : int(x))

    new_version = find_new_version(match_df)

    # 정확한 통계를 위해 가장 최신의 버전과 클래식 게임에 대한 데이터만 가져오자
    match_df = match_df.loc[(match_df['gameVersion']==new_version) & (match_df['gameMode']=='CLASSIC'), :]

    # 그 중에서도 이번 분석에서는 소환사의 협곡 솔로 랭크와 팀 랭크 게임만 사용한다.
    select_indices = (match_df['queueId']==420) | (match_df['queueId']==440)

    match_df = match_df.loc[select_indices, :].reset_index(drop=True)

    # DataFrame 내의 리스트들이 파일로 저장되었다가 불러지는 과정에서 문자로 인식됨
    for column in ['teams', 'participants', 'participantIdentities']:
        match_df[column] = match_df[column].map(lambda v: eval(v)) # 각 값에 대해 eval 함수를 적용
    return match_df


def find_new_version(df):
    match_df = df.copy()
    version_list = list(map(lambda x: x if type(x) == str else '0'  , Counter(match_df["gameVersion"]).keys()))
    int_version = list(map(lambda x: int(x.replace(".","")) , version_list))
    num = int_version.index(max(int_version))
    result = version_list[num]
    return result


match_df  = modifiy_match_df_original(match_df)
match_df.shape
#match_df2 = modfify_match_df(match_df2)
#match_df.to_csv("match_df.csv")


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
def get_time_line_list(df, api_key_list):
    match_df = df.copy()
    match_timeline_list = []
    api_machine = api_box(api_key_list)
    for game_id in tqdm(match_df['gameId']): # 각 게임 아이디마다 요청
        api_url = 'https://kr.api.riotgames.com/lol/match/v4/timelines/by-match/{}?api_key={}'.format(game_id , api_machine.current_api_key)
        r = requests.get(api_url)
        while r.status_code!=200: # 요청 제한 또는 오류로 인해 정상적으로 받아오지 않는 상태라면, 3초 간 시간을 지연
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


match_time_list = get_time_line_list(match_df, api_key_list)





len(match_time_list)

# f = open('MatchTimelineData.pickle', 'wb') # 리스트 안의 데이터프레임 형태이므로 바이너리 코드로 저장하기 위함임
# pickle.dump(match_time_list, f)
# f.close()

# 블랙리스트 되서 또 11788개의 데이터만 받아왔음

f = open('MatchTimelineData.pickle', 'rb')
match_timeline_list = pickle.load(f)




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

spell = spell_general_info()
spell

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
            lane["JUNGLE_{}".format(str(team))] = jungle_participant
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

def my_load_match_df():
    match_df = pd.read_csv("match_df.csv", index_col = 0)
    for column in ['teams', 'participants', 'participantIdentities']:
        match_df[column] = match_df[column].map(lambda v: eval(v))
    return match_df




len(match_time_list)
match_df.shape

lane_matching_df = participants_for_lanes(match_df, match_time_list)
lane_matching_df


######################## -----                 start from here  ----------------------  ##########################################
#lane_matching_df.to_csv("lane_matching.csv")
#lane_matching_df = pd.read_csv("lane_matching.csv", index_col = 0)
#match_df = my_load_match_df()

# f = open('MatchTimelineData.pickle', 'rb')
# match_timeline_list = pickle.load(f)
# f.close()
#df[df.isnull().any(axis=1)]
# lane_matching_df[lane_matching_df.isnull().any(axis=1)]

lane_matching_df

def modify_lane_matching_df(df):
    lane_matching_df = df.copy()
    lane_matching_df = lane_matching_df.reset_index(drop = True)
    lane_matching_df = lane_matching_df.dropna()
    lane_matching_df = lane_matching_df.astype(int)
    return lane_matching_df

# 널 값 처리 및 전부 정수로 변환
lane_info = modify_lane_matching_df(lane_matching_df)
lane_info
# 라인 매칭 데이터 프레임과 게임 정보 데이터프레임 합치고 데이터프레임 생성
def merge_lane_info_to_match_info(match, lane):
    match_info = match.copy()
    lane_info = lane.copy()
    merged = pd.merge(match_info,lane_info, on = "gameId" , how ="inner")
    return merged

merged_info = merge_lane_info_to_match_info(match_df, lane_info)
merged_info.columns
match_df.columns
lane_info.columns

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
merged_info_saving = merged_info.copy()

merged_info = modify_merged_info(merged_info)

merged_info

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
        5/0



merged_info = get_win_loss_col(merged_info)
merged_info.shape


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

merged_added = get_champion_sumId_cols(merged_info)

merged_added

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
api_key_list = [api_key1, api_key2]

all_lanes =["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]


merged_added.columns

def coerce_df_columns_to_numeric(df):
    columns = []
    for column in df.columns.tolist():
        if is_numeric_dtype(df[column]):
            columns.append(column)
    column_list = columns
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
    return df
merged_added = coerce_df_columns_to_numeric(merged_added)


def del_nan_merge_large(df):
    merged_large = df.copy()
    merged_large = merged_large.dropna()
    merged_large = coerce_df_columns_to_numeric(merged_large)
    return merged_large


merged_added= del_nan_merge_large(merged_added)
merged_added.to_csv("merged_large_del_na.csv")

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
'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key
path = 'https://kr.api.riotgames.com/'
new  = path +"lol/summoner/v4/summoners/by-name/{}?api_key={}".format(gm_df.summonerName.iloc[0],api_key)
r = requests.get(new)
r.json()
new
gm_df.head()

/lol/summoner/v4/summoners/{encryptedSummonerId}
/lol/summoner/v4/summoners/by-name/{summonerName}

merged_large.TOP100_sumID.iloc[0]
merged_added.iloc[0,:]
path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/{}?api_key={}".format(merged_added.TOP100_sumID.iloc[0], main_api_key  )
r= requests.get(path)
r.json()

path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}".format(merged_added.TOP100_sumName.iloc[0],main_api_key )
r= requests.get(path)
r.json()

path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}".format(merged_added.TOP100_sumName.iloc[0],api_key2 )
r= requests.get(path)
r.json()
merged_large = merged_added

lane_team = "TOP100"

merged_large.columns
i = 0


########################## not interchageable #####################################################
############################# but very interchangeable if you use name ############################

def get_summonerLevel(df, api_key_list ,lane_team):
    merged_large = df.copy()
    api_machine = api_box(api_key_list)
    path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
    merged_large["{}_summonerLevel".format(lane_team)] = None
    for i in tqdm(range(len(merged_large  ))):
        time.sleep(1)
        try:
            api_url = path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] ,  api_machine.current_api_key )
            r = requests.get(api_url)
            while r.status_code == 429:
                time.sleep(1)
                api_url = path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] ,  api_machine.switch() )
                r = requests.get(api_url)
            merged_large["{}_summonerLevel".format(lane_team)].iloc[i] = r.json()["summonerLevel"]
        except:
            print("there is something wrong wit this row")
            print(i)
    return merged_large


def get_summonerLevel_for_all_lanes(df,api_key_list,lanes):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    for lane in all_lanes:
        merged_added = get_summonerLevel(merged_added, api_key_list,lane)
    return merged_added




----------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% not interchangeable %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-----------------------------------------
$$$$$$$$$$$$$$$$$$$$$$$$$$$ but made it kinda working so moving on $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def get_win_los_rate_info(df, main_api_key, api_key_list, lane_team):
    merged_large = df.copy()
    api_machine  = api_box(api_key_list)
    path = "https://kr.api.riotgames.com/lol/league/v4/entries/by-summoner/{}?api_key={}"
    tempo_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
    temp_df = pd.DataFrame()
    api_urls = list(map(lambda sum: path.format(sum,main_api_key ), merged_large["{}_sumID".format(lane_team)]  ))
    if len(api_urls_first) != len(merged_large) or len(api_urls_second) != len(merged_large):
        print("created api urls do not have the same length as the df, aborting and causing an error on purpose")
        5 / 0
    for i in tqdm(range(len(api_urls_first))):
        time.sleep(1)
        try:
            api_url = api_urls[i]
            r = requests.get(api_url)
            while r.status_code == 429:
                tempo_api_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] , api_machine.switch())
                tempo_r = requests.get(tempo_api_url)
                tempo_id = tempo_r.json()["id"]
                api_url = path.format(tempo_id, api_machine.current_api_key)
                r = requests.get(api_url)
            temp_df[i] =  pd.Series(r.json()[0] )
        except:
            print("there is something wrong with this row")
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
            print("there is some thign wrong with the length, so aborting")
            5/0
        return merged_large

def get_win_los_rate_info_all_lanes(df,api_key_list,lanes):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    for lane in all_lanes:
        merged_added = get_top10avg_champ_detail_info(merged_added, api_key_list,lane)
    return merged_added










$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ not interchangeable but fk $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4

api_machine  = api_box(api_key_list)
api_machine.switch()
picked_champion_ids = merged_large["{}_champ".format(lane_team)].tolist()
picked_champion_ids

path = 'https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{}?api_key={}'
api_url = path.format(merged_large["{}_sumID".format(lane_team)].iloc[0] , main_api_key)
r = requests.get(api_url)
r.json()
path = 'https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{}?api_key={}'
api_url1 = path.format(merged_large["{}_sumID".format(lane_team)].iloc[0] , api_key1)
r1 = requests.get(api_url1)
r1.json()

tempo_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
tempo_api_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[0] , api_machine.switch() )
tempo_r = requests.get(tempo_api_url)
tempo_id = tempo_r.json()["id"]
tempo_id
api_url = path.format(tempo_id, api_machine.current_api_key)
r  = requests.get(api_url)
all_json = r.json()


len(all_json)

current_json = all_json[:5]
picked_id =  picked_champion_ids[i]
picked_id

championId_list = list(map(lambda x: x["championId"] , all_json))
picked_id in championId_list
championId_list.index(54)
all_json[-1]
all_json[-1]

picked_index  =  championId_list.index(picked_id)
all_json[picked_index]
all_json[-1]



def get_top10avg_champ_detail_info(df,main_api_key,  api_key_list, lane_team):
    merged_large = df.copy()
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

    if len(api_urls_first) != len(merged_large) or len(api_urls_second) != len(merged_large) or len(picked_champion_ids) != len(merged_large):
        print("created api urls do not have the same length as the df, aborting and causing an error on purpose")
        5 / 0

    for i in tqdm(range(len(api_urls_first))):
        try:
            time.sleep(1)
            api_url = api_urls_list[i]
            r= requests.get(api_url)

            trials = 0
            while r.status_code == 429:
                trials = trials + 1
                if trials == 70:
                    continue
                try:
                    tempo_api_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[0] , api_machine.switch() )
                    tempo_r = requests.get(tempo_api_url)
                    tempo_id = tempo_r.json()["id"]
                    api_url = path.format(tempo_id, api_machine.current_api_key)
                    r  = requests.get(api_url)
                except:
                    continue
                time.sleep(1)

            all_json = r.json()
            current_json = all_json[:5]
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
        except:
            print(i)
            print("unknown error occured")
            continue
    return merged_large


def champion_avg_detail_for_all_lanes(df,api_key_list,lanes):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    for lane in all_lanes:
        merged_added = get_top10avg_champ_detail_info(merged_added, api_key_list,lane)
    return merged_added

-------------------------------------
all_champs_list = merged_large["{}_champ".format(lane_team)].tolist()

all_champs_list[i]
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  last function !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

path = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/{}?api_key={}&champion={}&endIndex=5"


api_url  = path.format(merged_large["{}_accountId".format(lane_team)].iloc[i], main_api_key,all_champs_list[i] )

r = requests.get(api_url)
r.json()

tempo_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
tempo_

all_accountId_list = merged_large["{}_accountId".format(lane_team)].tolist()
all_accountId_list
all_champs_list = merged_large["{}_champ".format(lane_team)].tolist()
all_champs_list
api_url_list = list(map(lambda accountId, champion: path.format(accountId,main_api_key,champion),all_accountId_list,all_champs_list))
api_url_list

current_championId = all_champs_list[i]
current_championId
api_url = api_url_list[i]
r= requests.get(api_url)
r.json()

tempo_path = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}"
tempo_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] , api_machine.switch()  )
tempo_r = requests.get(tempo_url)
tempo_account = tempo_r.json()["accountId"]
api_url  = path.format(tempo_account , api_machine.current_api_key,  current_championId )
testing_r = requests.get(api_url)
testing_r.json()

r.json()
r = requests.get(api_url)
r.json()
merged_large.iloc[0,:]
tempo_account
current_dict = r.json()
current_dict

current_championId = all_champs_list[i]
api_url = api_url_list[i]
r= requests.get(api_url)
r.json()

current_dict = r.json()
current_dict

tempo_gameids = list(map(lambda x: x["gameId"], current_dict["matches"] ))
tempo_gameids
api_key_list
api_url
api_url = api_url_list[i]
r= requests.get(api_url)
r.json()
new_path = "https://kr.api.riotgames.com/lol/match/v4/matches/{}?api_key={}"
main_api_url = new.path()
new_api_url = new_path.format(each_game, api_machine.switch())
sr = requests.get(api_url)
sr.json()
each_game = tempo_gameids[0]
new_api_url = new_path.format(each_game, api_machine.switch())
sr = requests.get(new_api_url)
sr.json()



ma_api_url = new_path.format(each_game, main_api_key )
ma = requests.get(ma_api_url )
ma.json()

new_api_url = new_path.format(each_game, api_machine.switch())
new_r = requests.get(new_api_url)
new_r.json()
all_apis
all_apis = api_key_list
all_apis.append(main_api_key)
all_api_machine = api_box(all_apis)


new_api_url = new_path.format(each_game, all_api_machine.switch())
new_r = requests.get(new_api_url)
new_r.json()
new_current_data = new_r.json()
current_championId


game_info =  new_current_data["participants"]
participatn_index = list(map(lambda x: x["championId"] , game_info )).index(current_championId)

new_current_data

participatn_index
game_info[participatn_index]


a = -3
while a < 5:
    print(a)
    a = a+ 1
    try:
        b = 5 / a
        print(b)
    except:
        continue


merged_large.columns



api_key_list = [api_key1 , api_key2]
lane_team
merged_large.shape
import datetime
a = datetime.datetime.now()
merged_testing = very_detail_champ_info_and_history(merged_large, main_api_key , api_key_list , lane_team)
b = datetime.datetime.now()
print(b-a)

c = b -a
c +c
43 / 12


912 * 3 * 10 / 60 /60


np.zeros((5 )).mean()

merged_testing

def very_detail_champ_info_and_history(df , main_api_key, api_key_list, lane_team):
    merged_large = df.copy()
    api_machine = api_box(api_key_list)
    all_apis = api_key_list
    all_apis.append(main_api_key)
    all_api_machine = api_box(all_apis)
    path = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/{}?api_key={}&champion={}&endIndex=5"
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
        5/ 0
    for i in tqdm(range(len(api_url_list))):
        try:
            fked = 0
            #time.sleep(1)
            current_championId = all_champs_list[i]
            api_url = api_url_list[i]
            r= requests.get(api_url)
            if r.status_code == 404:
                merged_large["{}_total_champion_games".format(lane_team)].iloc[i] = 0
                continue
            trials = 0
            while r.status_code == 429:
                trials = trials + 1
                if trials == 50:
                    continue
                tempo_url = tempo_path.format(merged_large["{}_sumName".format(lane_team)].iloc[i] , api_machine.switch() )
                try:
                    tempo_r = requests.get(tempo_url)
                    tempo_account = tempo_r.json()["accountId"]
                except:
                    continue
                api_url  = path.format(tempo_account , api_machine.current_api_key,  current_championId )
                r = requests.get(api_url)
            if  r.status_code == 404:
                merged_large["{}_total_champion_games".format(lane_team)].iloc[i] = 0
                fked = 1
                #time.sleep(1)
            if fked == 1:
                continue


            current_dict = r.json()
            merged_large["{}_total_champion_games".format(lane_team)].iloc[i] = current_dict["totalGames"]

            tempo_gameids = list(map(lambda x: x["gameId"], current_dict["matches"] ))

            a = datetime.datetime.now()
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
            b = datetime.datetime.now()
            c2 = b - a

            for each_game in tempo_gameids:
                try:
                    #time.sleep(1)
                    new_api_url = new_path.format(each_game, all_api_machine.switch())
                    new_r = requests.get(new_api_url)
                    trials = 0
                    while new_r.status_code == 429:
                        trials = trials + 1
                        if trials == 50:
                            continue
                        try:
                            new_api_url = new_path.format(each_game , all_api_machine.switch())
                            new_r = requests.get(new_api_url)
                        except:
                            continue
                        #time.sleep(1)
                    new_current_data = new_r.json()
                    game_duration = float(new_current_data["gameDuration"] / 100)
                    game_info =  new_current_data["participants"]
                    participant_index = list(map(lambda x: x["championId"] , game_info )).index(current_championId)
                    a = datetime.datetime.now()
                    win_or_loss.append(float(game_info[participant_index]["stats"]["win"]))
                    kills.append(float(game_info[participant_index]["stats"]["kills"]/ game_duration))
                    deaths.append(float(game_info[participant_index]["stats"]["deaths"] / game_duration))
                    assists.append(float(game_info[participant_index]["stats"]["assists"] / game_duration))
                    total_damage.append(float(game_info[participant_index]["stats"]["totalDamageDealt"] / game_duration))
                    damage_to_champ.append(float(game_info[participant_index]["stats"]["totalDamageDealtToChampions"] / game_duration))
                    damage_to_objects.append(float(game_info[participant_index]["stats"]["damageDealtToObjectives"] / game_duration))
                    champ_level.append(float(game_info[participant_index]["stats"]["champLevel"] / game_duration))
                    gold_earned.append(float(game_info[participant_index]["stats"]["goldEarned"] / game_duration))
                    vision_score.append(float(game_info[participant_index]["stats"]["visionScore"]))
                    minionskilled.append(float(game_info[participant_index]["stats"]["totalMinionsKilled"] / game_duration))
                    neutral_minions_killed.append(float(game_info[participant_index]["stats"]["neutralMinionsKilled"] / game_duration))
                    wards_placed.append(float(game_info[participant_index]["stats"]["wardsPlaced"] / game_duration))
                    wards_killed.append(float(game_info[participant_index]["stats"]["wardsKilled"] / game_duration))
                    damage_taken.append(float(game_info[participant_index]["stats"]["totalDamageTaken"] / game_duration))
                    b = datetime.datetime.now()
                    c3 = b - a

                except:
                    continue
            a = datetime.datetime.now()
            avg_win_or_loss = sum(win_or_loss) / len(tempo_gameids)
            avg_kills =  sum(kills) /len(tempo_gameids)
            avg_deaths = sum(deaths)/len(tempo_gameids)
            avg_assists = sum(assists)/len(tempo_gameids)
            avg_total_damage = sum(total_damage)/len(tempo_gameids)
            avg_damage_to_champ = sum(damage_to_champ)/len(tempo_gameids)
            avg_damage_to_objects = sum(damage_to_objects)/len(tempo_gameids)
            avg_champ_level = sum(champ_level)/len(tempo_gameids)
            avg_gold_earned = sum(gold_earned)/len(tempo_gameids)
            avg_vision_score = sum(vision_score)/len(tempo_gameids)
            avg_minionskilled = sum(minionskilled)/len(tempo_gameids)
            avg_neutral_minions_killed = sum(neutral_minions_killed)/len(tempo_gameids)
            avg_wards_placed = sum(wards_placed)/len(tempo_gameids)
            avg_wards_killed = sum(wards_killed)/len(tempo_gameids)
            avg_damage_taken = sum(damage_taken)/len(tempo_gameids)

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
            b = datetime.datetime.now()
            c4 = b - a
            print(c2+c3+c4)

        except Exception as e:
            print("{}".format(e))
            continue
    return merged_large


merged_large.columns


participatn_index
participant_index = list(map(lambda x: x["championId"] , game_info )).index(current_championId)
new_api_url = new_path.format(each_game , all_api_machine.switch())
new_r = requests.get(new_api_url)
new_current_data = new_r.json()
game_info =  new_current_data["participants"]

float(game_info[participant_index]["stats"]["win"])

sum([1,3]) /3

float(True)

game_info[participant_index]["stats"]
float(game_info[participant_index]["stats"]["totalDamageTaken"] / game_duration)
float(game_info[participant_index]["stats"]["visionScore"])

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

new_apis = api_box(api_key_list)
new_apis.switch()




def very_detail_champ_info_and_history_for_all_lanes(df,api_key_list,lanes):
    #all_lanes = ["TOP100","JUNGLE100","MID100","ADC100","SUPPORT100","TOP200","JUNGLE200","MID200","ADC200","SUPPORT200"]
    merged_added = df.copy()
    for lane in all_lanes:
        merged_added = very_detail_champ_info_and_history(merged_added, api_key_list,lane)
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

-----------------------------------------
###########################################



def change_champID_to_champName(df ,general_champ_df ,lanes,):
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

testing = change_champID_to_champName(merged_added, champ_info , all_lanes)
testing
