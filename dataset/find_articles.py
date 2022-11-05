import csv
import re
import html
from datetime import datetime

# determine total number of images found
# save all articles that probably have images with alt descriptions

def status_update(num_of_articles, start_time, total_num_of_articles=20777127, progress_bar=True):
    # total number of articles: 20777127
    if num_of_articles % 5000 == 0:
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

# load wikipedia xml database
f = open('../../../../../../mnt/fs5/ekreiss/datasets/Wikipedia/enwiki-20201201-pages-articles-multistream.txt', 'r')

num_of_articles = 0
contains_img = 0
potential_descr = 0
relevant_article = False
writer_count = 0
current_article = ""
csv_data = []

start_time = datetime.now()

# initialize csv file to save extracted image file names and corresponding description and caption
with open('../../../../../../mnt/fs5/ekreiss/datasets/Wikipedia/articleextr.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows([["article"]])
    # go through each line in wikipedia xml file
    for line in f:
        line = html.unescape(line)
        # note if it marks the start of a new article
        if re.search('<title>.*</title>', line):
            # if the previous article contained a potential description, add article to csv file
            if relevant_article:
                csv_data.append([current_article])
                writer_count += 1
                if writer_count >= 5000:
                    print("writing")
                    writer.writerows(csv_data)
                    writer_count = 0
                    csv_data = []
            # initialize new article
            current_article = line[line.index('<title>')+7:line.index('</title>')]
            num_of_articles += 1
            relevant_article = False
            status_update(num_of_articles, start_time, progress_bar=True)
        # discard articles that start with "Wikipedia:" since they often have specialized symbols in images and descriptions/captions are highly conventionalized for the wikipedia usage
        # line should not be commented out
        # line is interesting if it contains an image file and the "alt" argument
        elif (not current_article.startswith("Wikipedia:")) and re.search("\.svg|\.png|\.jpg|\.jpeg|\.tif|\.xcf", line, re.IGNORECASE) and not line.startswith("<!--"):
            contains_img += 1
            # if alt tag in line
            if re.search("[Aa]lt *?[=:]", line):
                # determine length of text after alt -- filter out empty strings
                alt_proxi = re.search('[Aa]lt *?[=:](.*)\|', line)
                if not alt_proxi and re.search('[Aa]lt *?[=:](.*)\]', line):
                    alt_proxi = re.search('[Aa]lt *?[=:](.*)\]', line)
                elif not alt_proxi:
                    alt_proxi = re.search('[Aa]lt *?[=:](.*)$', line)
                if len(alt_proxi.group(1)) > 2:
                    potential_descr += 1
                    relevant_article = True
            
    writer.writerows(csv_data)

csv_file.close()

f.close()

print('num_of_articles: ', num_of_articles)
print('contains_img: ', contains_img)
print('potential_descr: ', potential_descr)