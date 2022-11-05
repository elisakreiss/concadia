import pandas as pd
import json
import numpy as np

# randomize function
def randomize(data, col_to_scramble):
  # separate by splits
  data_train = data[data['split']=="train"].copy(deep=True)
  data_test = data[data['split']=="test"].copy(deep=True)
  data_val = data[data['split']=="val"].copy(deep=True)
  # randomize within splits
  data_train[col_to_scramble] = np.random.permutation(data_train[col_to_scramble].values)
  # we leave val and test set unscrambled for a more intuitive comparison to the non-scrambled condition
  # data_test[col_to_scramble] = np.random.permutation(data_test[col_to_scramble].values)
  # data_val[col_to_scramble] = np.random.permutation(data_val[col_to_scramble].values)

  # merge splits
  data_full = data_train.append(data_test)
  data_full = data_full.append(data_val)

  data_full.to_csv('/mnt/fs5/ekreiss/datasets/Wikipedia/final_data_random' + col_to_scramble + '.csv', index=False)

  # revert to original format
  data_scrambled = {'images': data_full.to_dict('records')}

  with open('/mnt/fs5/ekreiss/datasets/Wikipedia/wiki_split_random' + col_to_scramble + '.json', 'w') as outfile:
      json.dump(data_scrambled, outfile, indent=4)

# import json and convert to pandas dataframe
with open('/mnt/fs5/ekreiss/datasets/Wikipedia/wiki_split.json') as f:
  js_obj = json.load(f)
to_pd = js_obj['images']
data = pd.DataFrame.from_dict(to_pd)
print(data.columns.tolist()) # ['article_id', 'filename', 'orig_filename', 'description', 'caption', 'context', 'split']

data.to_csv('/mnt/fs5/ekreiss/datasets/Wikipedia/final_data.csv', index=False)
  
randomize(data, 'filename')
randomize(data, 'description')
randomize(data, 'caption')
randomize(data, 'context')