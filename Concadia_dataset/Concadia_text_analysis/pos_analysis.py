import json
import nltk
import pandas as pd

print("loading data")
# import text data json
with open('wiki_split.json', 'r') as j:
    data = json.load(j)

# download required nltk packages
# required for tokenization
nltk.download('punkt')
# required for parts of speech tagging
nltk.download('averaged_perceptron_tagger')

max_len = 50
data_count = 0
all_descr_tags = []
all_caption_tags = []

print(len(data['images']))

print("sorting data")
for img_datum in data['images']:
    if data_count % 10000 == 0:
        print(data_count)
    data_count += 1

    if (len(img_datum['description']['tokens']) > max_len) or (len(img_datum['caption']['tokens']) > max_len):
        continue
    
    descr_tokens = nltk.word_tokenize(img_datum['description']['raw'])
    caption_tokens = nltk.word_tokenize(img_datum['caption']['raw'])

    descr_tagged = nltk.pos_tag(descr_tokens)
    caption_tagged = nltk.pos_tag(caption_tokens)

    descr_tags = [token[1] for token in descr_tagged]
    caption_tags = [token[1] for token in caption_tagged]

    all_descr_tags.extend(descr_tags)
    all_caption_tags.extend(caption_tags)

print("writing csv")
d = {'descr_tags': all_descr_tags}
df = pd.DataFrame(data=d)
df.to_csv('pos_tags_descr.csv')
d = {'caption_tags': all_caption_tags}
df = pd.DataFrame(data=d)
df.to_csv('pos_tags_caption.csv')
