import pandas as pd
import os
from nlgeval import NLGEval
import csv

nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'EmbeddingAverageCosineSimilarity', 
                                    'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 
                                    'GreedyMatchingScore'])  # loads the models


csv_full = [['label_cond', 'context_cond', 'randomized', 'nlg_type', 'context_encoder_type', 'run_id',
            'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']]

for subdir in os.listdir('.'):
    if not os.path.isdir(subdir):
        continue
    print(subdir)
    label_cond, context_cond, randomized, nlg_type, context_encoder_type = subdir.split('_')
    for fp in os.listdir(subdir):
        print('    ' + fp)
        data = pd.read_csv(os.path.join(subdir, fp))
        run_id = fp.split('.')[0]
        references = [data['label_true'].tolist()]
        hypothesis = data['label_generated'].tolist()
        metrics_dict = nlgeval.compute_metrics(references, hypothesis)
        csv_data = [label_cond, context_cond, randomized, nlg_type, context_encoder_type, run_id,
                    metrics_dict["Bleu_1"], metrics_dict["Bleu_2"], metrics_dict["Bleu_3"], metrics_dict["Bleu_4"],metrics_dict["CIDEr"], metrics_dict["METEOR"], metrics_dict["ROUGE_L"]]
        csv_full.append(csv_data)

filename = 'full_performance_data.csv'

with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    # # writing the fields 
    # csvwriter.writerow(fields) 
    # writing the data rows 
    csvwriter.writerows(csv_full)
    
