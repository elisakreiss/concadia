import argparse
import os
import csv
import json

EVAL_FN = 'pred.vinvl_data.test.best.beam5.max52.odlabels.eval.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default='/home/ubuntu/s3-drive/concadia_oscar_runs/', type=str, required=False,
                        help="`output_dir` in `oscar/run_captioning.py`.")

    args = parser.parse_args()

    csv_full = [['label_cond', 'context_cond', 'randomized', 'run_id',
                'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']]

    for run_config in os.listdir(args.run_dir):
        subdir = os.path.join(args.run_dir, run_config)
        if not os.path.isdir(subdir):
            continue
        label_cond, context_cond, randomized = run_config.split('_')
        for run_id in os.listdir(subdir):
            test_eval_metrics_fp = os.path.join(subdir, run_id, 'test_eval', EVAL_FN)
            if not os.path.isfile(test_eval_metrics_fp):
                print(f'Run {run_id} with config {run_config} does not have test eval.')
                continue
            with open(test_eval_metrics_fp, 'r') as f:
                metrics_dict = json.load(f)
            csv_data = [label_cond, context_cond, randomized, run_id,
                        metrics_dict["Bleu_1"], metrics_dict["Bleu_2"], metrics_dict["Bleu_3"], metrics_dict["Bleu_4"],metrics_dict["CIDEr"], metrics_dict["METEOR"], metrics_dict["ROUGE_L"]]
            csv_full.append(csv_data)

    filename = 'full_performance_data.csv'

    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the data rows 
        csvwriter.writerows(csv_full)
    
if __name__ == "__main__":
    main()