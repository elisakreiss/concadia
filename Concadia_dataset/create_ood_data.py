import json
import pandas as pd

# import Concadia json and convert to pandas dataframe
with open('wiki_split.json') as f:
  js_obj = json.load(f)
to_pd = js_obj['images']
data_wiki = pd.DataFrame.from_dict(to_pd)
print("Concadia: ", data_wiki.columns.tolist())
print(data_wiki.shape[0])

# # import GoodNews json and convert to pandas dataframe
# with open('../GoodNews/data/news_dataset.json') as f:
#   js_obj = json.load(f)
# to_pd = js_obj['images']
# data_goodnews = pd.DataFrame.from_dict(to_pd)
# print("Goodnews: ", data_goodnews.columns.tolist())
# print(data_goodnews.shape[0])
# print(data_goodnews['article'][0])

# paragraphs_goodnews = pd.read_csv('GoodNewsParagraphs.csv')
# print(paragraphs_goodnews.shape[0])

# import GoodNews json and convert to pandas dataframe
with open('GoodNewsParagraphs.json') as f:
  js_obj = json.load(f)
to_pd = js_obj
data_goodnews = pd.DataFrame.from_dict(to_pd)
print("Goodnews: ", data_goodnews.columns.tolist())
print(data_goodnews.shape[0])
# print(data_goodnews['paragraphs'][0])

# dicts = []
# for par in paragraphs_goodnews['paragraphs']:
#   print(par)
#   dict_entry = json.loads(par)
#   dicts.append(dict_entry)

data_wiki['ood'] = data_goodnews['paragraphs']
print(data_wiki[['ood','caption']])

# revert to original format
data_with_ood = {'images': data_wiki.to_dict('records')}

with open('wiki_split_ood.json', 'w') as outfile:
    json.dump(data_with_ood, outfile, indent=4)
