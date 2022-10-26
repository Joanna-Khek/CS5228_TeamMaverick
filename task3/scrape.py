import pandas as pd
import requests
import time

from selenium import webdriver
from selenium.webdriver.common.by import By

def get_hospital_names():
    list_of_elements = driver.find_elements(By.XPATH, "//span[@class='app_ment']")
    for elem in list_of_elements:
        hospitals.append(elem.text)
        
def get_coordinates(address):
    req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    resultsdict = eval(req.text)
    if (len(resultsdict["results"]) >0):
        lat.append(resultsdict["results"][0]["LATITUDE"])
        lng.append(resultsdict["results"][0]["LONGITUDE"])
    else:
        lat.append("Not Found")
        lng.append("Not Found")
        
if __name__ == "__main__":
    
    url = "https://www.healthhub.sg/directory/hospitals"

    chrome_options = webdriver.ChromeOptions()
    driver = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    driver.get(url)

    print("Scraping hospital names...")
    # scrape hospital name
    hospitals = []

    get_hospital_names()
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.find_elements(By.XPATH, "//a[@id='ctl00_ctl36_g_db0843df_d81d_4c3f_b3b0_3ad10493dba6_ctl00_PaginationMain_RepeaterPaging_ctl01_Pagingbtn']")[0].click()
    time.sleep(1)
    get_hospital_names()

    hospital_name = []
    for name in hospitals:
        hospital_name.append(name.split("\n")[0])
        
    print("Querying for map coordinates...")
    # map to get coordinates
    lat = []
    lng = []

    for addr in hospital_name:
        get_coordinates(addr)
        
    df = pd.DataFrame()
    df["name"] = hospital_name
    df["lat"] = lat
    df["lng"] = lng
    df.to_csv("../raw_data/additional-data/sg-hospital-data.csv", index=0)