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

    try:
        merged_info = modify_merged_info(merged_info)
    except Exception as e:
        print("modify_merged_info error : {}".format(e))
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


    pymysql.install_as_MySQLdb()
    host = "192.168.0.181"
    db_name = "lolpred"
    user_name = "root"
    password = "123"
    port = 3306
    db_type = "mysql"
    connect_db = connect_sql(host,db_name,user_name,password,port, db_type)
    connect_db.insert_df(final_final , "grandmaster_0805")
