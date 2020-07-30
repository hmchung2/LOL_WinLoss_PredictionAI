import pandas as pd
import inspect
import time
import json
from pprint import pprint
import xml.etree.ElementTree as ET
import argparse
import urllib.request
import numpy
import requests

#ServiceKey = 'RGAPI-e8e4a465-e3f1-47f9-9752-36809b966eca'

# url ="https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/Doublelift?api_key=" #RGAPI-YOUR-API-KEY
# url2 = "https://kr.api.pvp.net/api/lol/kr/v1.4"
# request = urllib.request.Request(url +ServiceKey)
# response = urllib.request.urlopen(request)
# rescode = response.getcode()
# response_body = response.read()
# response_body
# dict  = json.loads(response_body.decode('utf-8'))
# dict
api_key = 'RGAPI-eb28783d-51f2-4ce5-b509-89c44241532c' # Key를 갱신하여야 한다



def show_grandmaster_info(api_key):
    grandmaster = 'https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key=' + api_key
    r = requests.get(grandmaster)
    league_df = pd.DataFrame(r.json())
    league_df.reset_index(inplace=True)
    league_entries_df = pd.DataFrame(dict(league_df['entries'])).T
    league_df = pd.concat([league_df, league_entries_df], axis=1)
    league_df = league_df.drop(['index', 'queue', 'name', 'leagueId', 'entries', 'rank'], axis=1)
    return league_df



gm_df =show_grandmaster_info(api_key = api_key)
gm_df




def show_info(api_key, tier, division, page = 1):
    if tier.isupper() != True:
        raise ValueError('please write tier in upper case')
    roman_num = {'1':'I' , '2':'II', '3':'III', '4':'IV'}
    division_roman = roman_num[str(division)]
    query = 'https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/'+ tier+ '/'+ division_roman + '?page='+ str(page) +'&api_key=' + api_key
    r = requests.get(query)
    league_df = pd.DataFrame(r.json())
    return league_df

show_info(api_key,'GOLD',1, 1)


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

league_df_short = gm_df.head().copy()

league_df3 = df_summoner_accountid(league_df_short, api_key)



league_df3


i = 0
season = str(13)
match0 = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + league_df3['account_id'].iloc[i]  +'?season=' + season + '&endIndex=5&api_key=' + api_key
a = requests.get(match0).json()
a.keys()
len(a['matches'])
a



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
    return match_info_df


match_info_df =  accountID_to_matchINFO(league_df3 = league_df3, endIndex=5, api_key= api_key)




match_info_df.shape
match_info_df
match_info_df2  = match_info_df.iloc[:20,:]

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




match_fin = game_id_to_match_detail(match_info_df2, api_key)
from collections import Counter
Counter(match_fin["queueId"])



match_fin.shape
match_fin.shape
match_fin.columns
match_fin['participantIdentities'][0]

match_fin



one_game = match_fin['participants'][2]


one_game_lane = []
one_game_role = []
for person in one_game:
    a = person['timeline']['lane']
    b = person['timeline']['role']
    one_game_role.append(b)
    one_game_lane.append(a)


one_game_role

one_game_lane







def show_grandmaster_info(api_key):
    grandmaster = 'https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key=' + api_key
    r = requests.get(grandmaster)
    league_df = pd.DataFrame(r.json())
    league_df.reset_index(inplace=True)
    league_entries_df = pd.DataFrame(dict(league_df['entries'])).T
    league_df = pd.concat([league_df, league_entries_df], axis=1)
    league_df = league_df.drop(['index', 'queue', 'name', 'leagueId', 'entries', 'rank'], axis=1)
    return league_df



api_key

x = league_df3['summonerId'][0]
league_df3.iloc[0,:]


api_url = 'https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/'  + x + "?api_key=" +api_key

api_url
r = requests.get(api_url)
data = r.json()
data

api_url = 'https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/' + x +'/by-champion/38?api_key=' + api_key
r = requests.get(api_url)
data= r.json()
data



def summoner_id_to_personal_info(api_key):



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some info.')
    parser.add_argument('-k', '--apikey',dest = "api_key",type = str, default = 'RGAPI-e8e4a465-e3f1-47f9-9752-36809b966eca')
    args = parser.parse_args()
    result = show_grandmaster_info(args.api_key)
    print(result)




# api_key = ServiceKey
# grandmaster = 'https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key=' + api_key
# r = requests.get(grandmaster)
# league_df = pd.DataFrame(r.json())
# league_df.reset_index(inplace=True)
# league_entries_df = pd.DataFrame(dict(league_df['entries'])).T
# league_df = pd.concat([league_df, league_entries_df], axis=1)
# league_df = league_df.drop(['index', 'queue', 'name', 'leagueId', 'entries', 'rank'], axis=1)
# league_df
