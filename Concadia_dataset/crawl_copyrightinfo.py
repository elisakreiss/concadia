import re
import pandas as pd
import urllib.request
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup

def status_update(num_of_datapts, start_time, total_num_of_datapts, progress_bar=True):
    if num_of_datapts % 25 == 0:
        # Remaining time
        now = datetime.now()
        timespent_min = (now - start_time).total_seconds() / 60
        total_time = (timespent_min/num_of_datapts)*total_num_of_datapts
        remaining_time = divmod(total_time - timespent_min, 60)
        # Progress bar
        percent = ("{0:.2f}").format(100 * (num_of_datapts / float(total_num_of_datapts)))
        filled_length = int(100 * num_of_datapts // total_num_of_datapts)
        progress_bar = '█' * filled_length + '-' * (100 - filled_length)
        print('\r%s |%s| %s%% %s' % ('Progress: ', progress_bar, percent, 'Complete'), 
                    "; Remaining time: ", round(remaining_time[0]), 'hours', 
                    round(remaining_time[1]), 'minutes', end='\r')
    # Print New Line on Complete
    if num_of_datapts == total_num_of_datapts:
        print()

def get_info(lic_content):
    counter = 0
    info = lic_content
    text = lic_content
    while (len(text.split()) < 10) & (counter < 10):
        info = info.parent
        text = info.text
        counter += 1
    return(text)

# load csv containing articles with potential captions and descriptions
df = pd.read_csv('data_wfilenames.csv')
# df = df[600:800]

df['copyright'] = False
no_license = 0
rows = df.shape[0]
start_time = datetime.now()
for index, row in df.iterrows():
    if df['file_avail'][index]:
        status_update(index+1, start_time, rows)
        url = row['file_crawl']
        if re.search("\..{3,4}\..{3,4}$", url):
            img_name = re.sub("\..{3,4}$", "", url)
        else:
            img_name = url

        img_name = re.sub('.*/[0-9]{2,4}px--?', '', img_name)
        response = requests.get(
            url="https://commons.m.wikimedia.org/wiki/File:" + img_name,
        )
        if not (response.status_code == 200):
            img_name = re.sub('.*-[0-9]{2,4}px--?', '', img_name)
            response = requests.get(
                url="https://commons.m.wikimedia.org/wiki/File:" + img_name,
            )
        if not (response.status_code == 200):
            img_name = re.sub('.*wikipedia/commons/.{1,3}/.{1,3}/', '', img_name)
            response = requests.get(
                url="https://commons.m.wikimedia.org/wiki/File:" + img_name,
            )
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            license_frames = soup.find_all(text=re.compile(r'(This file is licensed under|public domain|protected by copyright|no known copyright restrictions|copyright holder|This work of art is free)'))
            divs = [("License info " + str(idx+1) + ": " + get_info(lic_content)) for idx, lic_content in enumerate(license_frames)]
            if not divs:
                no_license += 1
            df.loc[index,'copyright'] = "; ".join(divs)
        else:
            no_license += 1

print('images: ', index)
print("no_license: ", no_license)

df.to_csv('data_wcopyright.csv', index=False)
