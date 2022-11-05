import numpy as np
from PIL import Image
from nltk.tokenize import word_tokenize
import glob
import json
import random
import pandas as pd
import copy
import re
import os
import argparse
import sys

# import nltk
# nltk.download('punkt')

def format_text(text):
    # remove unicode characters
    ascii_text = text.encode("ascii", errors="ignore").decode()
    remove_nl  = ascii_text.replace("\n", "")
    cleaned_text = re.sub('\[[0-9]{1,4}\]','',remove_nl)
    cleaned_text = re.sub('[Ff]ile ?:','',cleaned_text)
    text_info = {}
    text_info['raw'] = cleaned_text
    text_info['tokens'] = word_tokenize(cleaned_text)
    return text_info

def label_invalid(label):
    if re.search('(refer to caption|alt(ernative)? text|see caption|\.(ogv|OGV)|\.(webm|WEBM))', label.lower()) or label.lower()=='text':
    # if re.search('(refer to caption|alt text|see caption)', label.lower()) or label.lower()=='text':
        invalid = True
    else:
        invalid = False
    return(invalid)

# parser = argparse.ArgumentParser(description='Define input parameters.')
# parser.add_argument('--randomize', type=str,
#                     help='img, description, caption, context, none')
# args = parser.parse_args()
# if not args.randomize in ['img', 'description', 'caption', 'context', 'none']:
#     sys.exit("invalid parser argument -- try img, description, caption, context, or none.")

df = pd.read_csv('/mnt/fs5/ekreiss/datasets/Wikipedia/data_wfilenames.csv')
print("original df, nr of rows: ", df.shape[0]) # 106677
# exclude data that we don't have an image from
data = copy.deepcopy(df[df['file_avail'] == True])
print("data only with available files, nr of rows: ", data.shape[0]) # 105960

data['invalid_label'] = True
# mark invalid labels
for idx, row in data.iterrows():
    desc_invalid = label_invalid(row['description_crawl'])
    # caption_invalid = label_invalid(row['caption'])
    data.loc[idx,'invalid_label'] = desc_invalid
# exclude data where labels are invalid
data = data[data['invalid_label'] == False]
print("data only with valid labels, nr of rows: ", data.shape[0]) # 102505

# add columns for description, caption and context length
data['descr_len'] = data['description_crawl'].str.len()
data['caption_len'] = data['caption_crawl'].str.len()
data['context_len'] = data['context'].str.len()
# exclude data where labels are shorter than 3 characters
data = data[(data['descr_len'] > 2) & (data['caption_len'] > 2) & (data['context_len'] > 2)] # 96947


print(data.columns.tolist())
print("data only with labels/context longer than 3 characters, nr of rows: ", data.shape[0])

data['img_repeat'] = data.duplicated(subset=['file_crawl'], keep=False)
print('datapoints where img repeated: ', sum(data['img_repeat'])) # 13948
data['same_label'] = data['description_crawl'] == data['caption_crawl']
print('datapoints where descr==caption: ', sum(data['same_label'])) # 7367
# have all images with repetitions or where descr=caption in training
data['split'] = np.where(data['img_repeat'] | data['same_label'], "train", "undecided")

print('datapoints certainly in train split: ', data[data['split']=='train'].shape[0]) # 20872

# make a by-article train/test/val split
train_samples = round(0.8 * data.shape[0])
test_samples = round(0.1 * data.shape[0])
val_samples = data.shape[0] - (train_samples + test_samples)

split_list = train_samples*['train'] + test_samples*['test'] + val_samples*['val']
random.shuffle(split_list)
print('number of splits: ', len(split_list)) # 96947

print("starting")
wiki_full_dataset = {}
wiki_full_dataset['images'] = []
left_over = []
new_datapoint = 0
for idx, row in data.iterrows():
# for row in data.itertuples():
    if idx % 10000 == 0:
        print('idx: ', idx)
    new_img = {}
    new_img['article_id'] = row['article']
    new_img['filename'] = str(row['row_id']) + ".jpg"
    new_img['orig_filename'] = row['file_crawl']
    new_img['description'] = format_text(row['description_crawl'])
    new_img['caption'] = format_text(row['caption_crawl'])
    new_img['context'] = format_text(row['context'])

    if bool(re.match('^ *$', new_img['description']['raw'])) or (not new_img['description']['tokens']):
        print(new_img['description'])
        continue
    if bool(re.match('^ *$', new_img['caption']['raw'])) or (not new_img['caption']['tokens']):
        print(new_img['caption'])
        continue
    if bool(re.match('^ *$', new_img['context']['raw'])) or (not new_img['context']['tokens']):
        print(new_img['context'])
        continue

    if not os.path.exists(os.path.join("/mnt/fs5/ekreiss/datasets/Wikipedia/wikicommons/resized/", new_img['filename'])):
        print(new_img['filename'], " doesn't exist")
        continue

    # images of the same article go either all in training or test
    split_in_article = data[data['article']==row['article']]['split'].tolist()
    if 'train' in split_in_article:
        new_img['split'] = 'train'
    elif 'test' in split_in_article:
        new_img['split'] = 'test'
    elif 'val' in split_in_article:
        new_img['split'] = 'val'
    else:
        new_img['split'] = split_list[0]
    # removes only first occurrence
    if new_img['split'] in split_list:
        split_list.remove(new_img['split'])
    else:
        left_over.append(new_img['split'])
    data.loc[idx,'split'] = new_img['split']

    wiki_full_dataset['images'].append(new_img)
    new_datapoint += 1

print(data['split'][:50].tolist())
print('number of datapoints in final dataset: ', new_datapoint) # 96918
print(len(split_list)) # 29
print(len(left_over)) # 0

with open('/mnt/fs5/ekreiss/datasets/Wikipedia/wiki_split.json', 'w') as outfile:
    json.dump(wiki_full_dataset, outfile, indent=4)



