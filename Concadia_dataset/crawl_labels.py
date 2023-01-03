import csv
import re
import html
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup

def status_update(num_of_articles, start_time, total_num_of_articles, progress_bar=True):
    # total number of articles
    if num_of_articles % 5 == 0:
        # Remaining time
        now = datetime.now()
        timespent_min = (now - start_time).total_seconds() / 60
        total_time = (timespent_min/num_of_articles)*total_num_of_articles
        remaining_time = divmod(total_time - timespent_min, 60)
        # Progress bar
        percent = ("{0:.2f}").format(100 * (num_of_articles / float(total_num_of_articles)))
        filled_length = int(100 * num_of_articles // total_num_of_articles)
        progress_bar = '█' * filled_length + '-' * (100 - filled_length)
        print('\r%s |%s| %s%% %s' % ('Progress: ', progress_bar, percent, 'Complete'), 
                    "; Remaining time: ", round(remaining_time[0]), 'hours', 
                    round(remaining_time[1]), 'minutes', end='\r')
    # Print New Line on Complete
    if num_of_articles == total_num_of_articles:
        print()

# load csv containing articles with potential captions and descriptions
# f = pd.read_csv('fileextr.csv')
f = pd.read_csv('articleextr.csv')
# f = f[:500]
articles = f.article.unique()

writer_count = 0
csv_data_crawl = []
file_extr_data = ""
number_of_articles = 0
total_num_of_articles = len(articles)

# initialize csv file to save extracted image file names and corresponding description and caption
with open('fileextr_crawl.csv', 'w') as csv_file_crawl:
    writer_crawl = csv.writer(csv_file_crawl)
    writer_crawl.writerows([["article", "file_crawl", "description_crawl", "caption_crawl", "context"]])

    start_time = datetime.now()
    # go through each line in csv file
    for article in articles:
        number_of_articles += 1
        status_update(number_of_articles, start_time, total_num_of_articles)
        # print("current_article: ", article)
        try:
            article_link = article.replace(" ", "_")
            response = requests.get(
                url="https://en.wikipedia.org/wiki/" + article_link,
            )
        except Exception as error_msg:
            print('article: ', article)
            print(error_msg)
            continue
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # to find all images, look in thumbs... we skip infobox
        thumbs = soup.findAll('div', class_="thumb")
        # print('-- found %d thumbs' % len(thumbs))
        for thumb in thumbs:
            thumb_imgs = thumb.findAll('img')
            if len(thumb_imgs) == 0:
                continue
            # there may be many images (rarely) in a caption.
            # just take the first one.
            thumb_img = thumb_imgs[0]
            if not thumb_img.has_attr('src'):
                continue
            thumb_img_url = 'https:%s' % thumb_img['src']
            thumb_captions = thumb.findAll('div', class_="thumbcaption")
            try:
                thumb_alt = thumb_img['alt']
            except:
                continue
            if len(thumb_captions) == 0 or len(thumb_alt) == 0:
                continue
            # there may be many captions (rarely)... just take the first one.
            thumb_caption = thumb_captions[0]
            thumb_caption = thumb_caption.text

            # print("IMG URL: ", thumb_img_url, "\n")
            # print("Caption: ", thumb_caption, "\n")
            # print("Alt: ", thumb_alt, "\n")

            # word_counter = 0
            word_min = 20
            paragraph = ''
            for sibling in thumb.next_siblings:
                if sibling.name != None:
                    paragraph = paragraph + " " + sibling.text
                    if(len(paragraph.split()) > word_min):
                        break;

            csv_data_crawl.append([article, thumb_img_url, thumb_alt, thumb_caption, paragraph])

        writer_count += 1
        if writer_count >= 5000:
            writer_crawl.writerows(csv_data_crawl)
            writer_count = 0
            csv_data_crawl = []

    writer_crawl.writerows(csv_data_crawl)

csv_file_crawl.close()
