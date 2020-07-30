# my_lol_accounts = {"midtuhon1005":"Incep&8133",
#                    "hmDeepLeague1005":"Incep&8133"}

# # sudo apt-get install chromium-chromedriver
# import os
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# import requests
#
main_api_key = "RGAPI-75e1131f-a7cf-4692-a2b6-b6ea1ee26100"
#
# current_path  = os.getcwd()
# chrome_path = os.path.join(current_path , "chromedriver")
#
# #############################################################################
# # 버전 문제 생기면 한번만 이렇게 실행 하고 새로 생긴 파일 현제 소스경로에 다시 옮기면 된다 ###
# #from webdriver_manager.chrome import ChromeDriverManager
# #driver = webdriver.Chrome(ChromeDriverManager().install())
#
#
# driver = webdriver.Chrome(executable_path = chrome_path)
# url = 'http://developer.riotgames.com/login'
# driver.get(url)
# username = driver.find_element_by_name('username')
# username.send_keys("midtuhon1005")
# password = driver.find_element_by_name('password')
# password.send_keys("Incep&8133")
#
#
# #driver.find_element_by_xpath(".//div[@class='into']/div[@rel='CA']" ).click()
#
# driver.find_element_by_xpath(".//button[@type='submit']").click()
#
# # driver.findElement(By.xpath("//tag[contains(text(),'usrn')]").getText();
# #
# # driver.findElement(By.xpath("//tag[contains(@AN,'usrn')]").getText();
# clicking = driver.find_elements_by_xpath(".//input[@type='submit']")[0]
# clicking
#
#
# url = 'http://developer.riotgames.com/login'
# data = {'username':'xxxxxxxxx','password':'yyyyyyyyy','challenge':'zzzzzzzzz','hash':''}
# # note that in email have encoded '@' like uuuuuuu%40gmail.com
#
# data= {"username":"midtuhon1005","password":"Incep&8133"}
# session = requests.Session()
# r = session.post(url,  data=data)
# a = r.history
# a
# for i in a:
#     print(i.url)
#
# with requests.Session() as session:
#     post = session.post("https://developer.riotgames.com", data=data)
#     r = session.get("https://developer.riotgames.com/login")
#     print(r.text)   #or whatever else you want to do with the request data!
#
# "100 requests" in r.text
#
#
#
#
# dir(r)
# print(r.url)
# for i in range(5):
#     print(r.url)
# a = open("./testing.txt" , "w+")
# a.write(r.url)
# a.close()
#
#
#
#
# payload = {'inUserName': 'USERNAME/EMAIL', 'inUserPass': 'PASSWORD'}
# url = 'http://www.locationary.com/home/index2.jsp'
# requests.post(url, data=payload)
