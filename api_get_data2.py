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
api_key = 'RGAPI-f7326880-0b44-49b7-aaf6-ad287f00d8f7'

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


gm_df.head()

league_df = gm_df
api_key

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

league_df




api_key = apikey
sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key
sohwan
r = requests.get(sohwan)
account_id = r.json()['accountId']
r.json()



league_df.iloc[0,-1]
league_df


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
