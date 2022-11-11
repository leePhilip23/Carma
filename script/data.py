import json

import requests
from requests.auth import HTTPBasicAuth
# import dill
from bs4 import BeautifulSoup
# from datetime import datetime
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML, fromstring, tostring
from util import *

creds = read_api_key()

LS_API_KEY = creds["here.api_key"]
auth = HTTPBasicAuth('apiKey', LS_API_KEY)
flow_url = f"https://data.traffic.hereapi.com/v7/flow"


destination = [33.0811809, -96.841015]
start = [32.9857, -96.7502]
page = requests.get(f'https://data.traffic.hereapi.com/v7/flow?apiKey={LS_API_KEY}'
                    f'&in=bbox:{min(start[1],destination[1])},{min(start[0],destination[0])},'
                    f'{max(start[1],destination[1])},{max(start[0],destination[0])}&locationReferencing=shape')

# print(f'https://data.traffic.hereapi.com/v7/flow?apiKey={LS_API_KEY}'
#                     f'&in=bbox:{min(start[1],destination[1])},{min(start[0],destination[0])},'
#                     f'{max(start[1],destination[1])},{max(start[0],destination[0])}&locationReferencing=olr')
print(page.json())
# file_name = "temp_traffic.json"
# with open(file_name, 'w') as file_object:  #open the file in write mode
#  json.dump(page.json(), file_object)
