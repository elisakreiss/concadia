import pandas as pd
import json
import numpy as np
import random
import shutil
import random
import os
import torch
import csv
import sys
sys.path.insert(0, '../../../models/03_xu2015/code')
import caption

gpu_id = 0
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

def load_model(label_cond):
  if label_cond == "description":
    run_id = "20210505_150530_descriptioncaptionnoneFalseFalseFalserevised"
    epoch = 29 # epoch 23
  else:
    run_id = "20210505_150623_captiondescriptionnoneFalseFalseFalserevised"
    epoch = 26 # epoch 20

  # Load specs
  specs = json.load(open('../../../models/03_xu2015/runs/run_' + run_id + '/' + 'specs.json', 'r'))
  data_location = specs['data_folder'].replace("../../../../../..","")
  # print(data_location)

  # Load model
  print("Loading model")
  checkpoint = torch.load("/mnt/fs5/ekreiss/qud_captioning/03_xu2015/runs/run_" + run_id + "/checkpoint_wikipedia_1_min_word_freq_epoch" + str(epoch) + ".pth.tar", map_location=str(device))
  decoder = checkpoint['decoder']
  decoder = decoder.to(device)
  decoder.eval()
  encoder = checkpoint['encoder']
  encoder = encoder.to(device)
  encoder.eval()

  # Load word map (word2ix)
  print("Loading word map")
  with open(data_location + "/WORDMAP_wikipedia_1_min_word_freq.json", 'r') as j:
    word_map = json.load(j)
  rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

  return([encoder, decoder, word_map, rev_word_map])


def generate(model, img_filename, context):
  [encoder, decoder, word_map, rev_word_map] = model

  img = "/mnt/fs5/ekreiss/datasets/Wikipedia/wikicommons/resized/" + img_filename

  # Generate model output
  # print("Generating model output")
  # try:
  seq = caption.labelwcontext_image_beam_search(encoder, decoder, img, context, word_map, 5, gpu_id=str(gpu_id))
  generated_label = " ".join([rev_word_map[ind] for ind in seq])
  generated_label = generated_label.replace('<start> ', '')
  generated_label = generated_label.replace(' <end>', '')
  return(generated_label)
  # except Exception as error_msg:
  #   print("An exception occurred.")
  #   print(error_msg)
  #   return("NA")

  

with open('/mnt/fs5/ekreiss/datasets/Wikipedia/wiki_split.json') as f:
  js_obj = json.load(f)
to_pd = js_obj['images']
data = pd.DataFrame.from_dict(to_pd)
print(len(data))

clean_data = data[(data['split']=="val")].copy(deep=True)
# print(len(clean_data))

randomlist = random.sample(range(0, len(clean_data)), 300)

sample = clean_data.iloc[randomlist].copy(deep=True)
# print(len(sample))

descr_model = load_model("description")
caption_model = load_model("caption")
# descr_encoder, descr_decoder, descr_wordmap, descr_rev_word_map = load_model("description")
# caption_encoder, caption_decoder, caption_wordmap, caption_rev_word_map = load_model("caption")

generated_descriptions = []
generated_captions = []
s_id = 0
for idx, row in sample.iterrows():
  gen_description = generate(descr_model, row['filename'], row['caption']['raw'])
  gen_caption = generate(caption_model, row['filename'], row['description']['raw'])
  # if (gen_description == "NA") or (gen_caption == "NA"):
  #   print(kasjdfhk)
  generated_descriptions.append(gen_description)
  generated_captions.append(gen_caption)
  shutil.copyfile('/mnt/fs5/ekreiss/datasets/Wikipedia/wikicommons/resized/' + row['filename'],
                  '../images/' + row['filename'])

sample['generated_descr'] = generated_descriptions
sample['generated_capt'] = generated_captions

# revert to original format
data_sample = {'images': sample.to_dict('records')}

with open('exp_sample.json', 'w') as outfile:
    json.dump(data_sample, outfile, indent=4)