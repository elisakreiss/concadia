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

# load csv containing articles with potential captions and descriptions
df = pd.read_csv('fileextr_crawl.csv')

df['row_id'] = np.arange(len(df))

df['file_avail'] = True
file_unavail = 0
rows = df.shape[0]
start_time = datetime.now()
for index, row in df.iterrows():
    if not re.search("\.gif$", row['file_crawl']):
        status_update(index+1, start_time, rows)
        filename = "wikicommons/wikiimages_raw/" + str(row['row_id']) + '.jpg'
        url = re.sub('/[0-9]{2,4}px', '/500px', row['file_crawl'])
        # img_name = re.sub(".*/","",row['file_crawl'])
        # if re.search("\..{3,4}\..{3,4}$", img_name):
        #     img_name = re.sub("\..{3,4}$", "", img_name)
        # else:
        #     img_name = img_name
        try:
            # urllib.request.urlretrieve(url, "wikicommons/wikiimages_raw/" + img_name)
            urllib.request.urlretrieve(url, filename)
        except Exception as error_msg:
            print("idx: ", index)
            print('issue with url: ', url)
            print('error_msg: ', error_msg)
            df.loc[index, 'file_avail'] = False
            file_unavail += 1
            continue
    else:
        df.loc[index, 'file_avail'] = False
        file_unavail += 1

print('images: ', index+1)
print("file_unavail: ", file_unavail)

df.to_csv('data_wfilenames.csv', index=False)
