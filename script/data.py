%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
import requests
#import dill
from bs4 import BeautifulSoup
#from datetime import datetime
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML, fromstring, tostring
page = requests.get('https://traffic.api.here.com/traffic/6.2/flow.xml?app_id=BLAH&app_code=BLAH2&bbox=39.039696,-77.222108;38.775208, -76.821107&responseattributes=sh,fc')
soup = BeautifulSoup(page.text, "lxml")
roads = soup.find_all('fi')