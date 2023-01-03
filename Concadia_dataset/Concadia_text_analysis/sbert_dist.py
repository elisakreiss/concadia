import json
import random
import copy
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("loading data")
# import text data json
with open('wiki_split.json', 'r') as j:
    data = json.load(j)

max_len = 50

descr_train_labels = []
descr_val_labels = []
descr_test_labels = []
caption_train_labels = []
caption_val_labels = []
caption_test_labels = []

train_data_count = 0

print("sorting data")
for img_datum in data['images']:
    # if train_data_count >= 10:
    #     break

    if (len(img_datum['description']['tokens']) > max_len) or (len(img_datum['caption']['tokens']) > max_len):
        continue

    descr_train_labels.append(img_datum['description']['raw'])
    caption_train_labels.append(img_datum['caption']['raw'])
    # train_data_count += 1

    # if img_datum['split'] == 'train':
    #     descr_train_labels.append(img_datum['description']['raw'])
    #     caption_train_labels.append(img_datum['caption']['raw'])
    #     train_data_count += 1
    # elif img_datum['split'] == 'val':
    #     descr_val_labels.append(img_datum['description']['raw'])
    #     caption_val_labels.append(img_datum['caption']['raw'])
    # elif img_datum['split'] == 'test':
    #     descr_test_labels.append(img_datum['description']['raw'])
    #     caption_test_labels.append(img_datum['caption']['raw'])

print("loading model")
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
SBert = SentenceTransformer('paraphrase-distilroberta-base-v1', device=device)

print("encoding data")
# descr_train_labels = torch.Tensor(descr_train_labels).to(device)
# caption_train_labels = torch.Tensor(caption_train_labels).to(device)
descr_train_emb = SBert.encode(descr_train_labels, show_progress_bar=False)
print("...description done")
caption_train_emb = SBert.encode(caption_train_labels, show_progress_bar=False)
print("...caption done")

# random_descr_train_emb = copy.deepcopy(descr_train_emb)
# random.shuffle(random_descr_train_emb)

# print("computing similarities")
# ordered_similarity = []
# random_similarity = []
# for i in range(len(descr_train_emb)):
#     descr = descr_train_emb[i].reshape(1, -1)
#     caption = caption_train_emb[i].reshape(1, -1)
#     rand_descr = random_descr_train_emb[i].reshape(1, -1)
#     o_sim = cosine_similarity(descr, caption)
#     ordered_similarity.append(o_sim[0][0].tolist())
#     r_sim = cosine_similarity(rand_descr, caption)
#     random_similarity.append(r_sim[0][0].tolist())

# print("ordered similarity")
# print(ordered_similarity)
# print("random similarity")
# print(random_similarity)

print("computing similarities")
pairwise_similarities=cosine_similarity(descr_train_emb, caption_train_emb)

print("sorting similarities")
corresponding_sim = []
other_sims = []
other_sim = []
for i,sim_vector in enumerate(pairwise_similarities):
    if i % 500 == 0:
        print(i)
    # correct match similarity
    corresponding_sim.append(sim_vector[i])
    # full non-match similarity list
    sim_list = sim_vector.tolist()
    sim_list.pop(i)
    # other_sims.append(sim_list)
    # random non-match similarity
    randint = random.randint(0,len(sim_list)-1)
    other_sim.append(sim_list.pop(randint))
# print(corresponding_sim)
# print(other_sims)
# print(other_sim)

print("writing csv")
# d = {'match_sim': corresponding_sim, 'nonmatch_sim': other_sim, 'nonmatch_sim_full': other_sims}
d = {'match_sim': corresponding_sim, 'nonmatch_sim': other_sim}
df = pd.DataFrame(data=d)
df.to_csv('sbert_sim.csv')

# print(pairwise_similarities)
