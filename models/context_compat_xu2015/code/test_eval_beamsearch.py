import json
import random
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import os
import caption
import torch
import matplotlib.patches as patches
import csv
from transformers import AutoTokenizer
from tqdm import tqdm

import wandb

def sample_output(run_id, data, checkpoint_path, label, context, nlg_type, context_encoder_type, blank_context, randomized, beam_size=5):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()
    except Exception as error_msg:
        print("Error occurred in loading checkpoint: " + error_msg)
        return


    output_dir = os.path.join('../paper_runs_perf', f"{label}_{'none' if blank_context else context}_{randomized}_{nlg_type}_{context_encoder_type}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    beamsearch_perf = [['run_id', 'beam_size', 'img', 'label_true', 'context_true', 'label_generated']]

    context_encoder_path = 'bert-base-uncased'
    if context_encoder_type == 'roberta':
        context_encoder_path = 'roberta-base'
    context_tokenizer = AutoTokenizer.from_pretrained(context_encoder_path)

    for img_data in tqdm(data):

        raw_label = img_data[label]['raw']

        # Generating model output
        try:
            image_fn = os.path.join(args.data_dir, "wikicommons/resized", img_data['filename'])
            if context == "none": # should not be executed
                seq, _ = caption.label_image_beam_search(encoder, decoder, image_fn, word_map, beam_size, gpu_id=args.gpu_id)
            else:
                seq = caption.labelwcontext_image_beam_search(encoder, decoder, context_tokenizer, image_fn, img_data[context]['raw'], word_map, blank_context, beam_size, gpu_id=args.gpu_id)
            # generated label
            generated_label = " ".join([rev_word_map[ind] for ind in seq])
            generated_label = generated_label.replace('<start> ', '')
            generated_label = generated_label.replace(' <end>', '')
            sample = [run_id, beam_size, img_data['filename'], raw_label, img_data[context]['raw'], generated_label]
            beamsearch_perf.append(sample)
        except Exception as error_msg:
            print("An exception occurred.")
            print(error_msg)

    filename = os.path.join(output_dir, f"{run_id}.csv")

    with open(filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerows(beamsearch_perf)


if __name__ == '__main__':
    # Initialize parameters
    parser = argparse.ArgumentParser(description='Define input parameters.')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='beam size')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='If cuda available, which device should be used?')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--data_dir', type=str,
                        default='/home/ubuntu/s3-drive/data',
                        help="Where data for model training and eval is stored")
    parser.add_argument('--runs_dir', type=str,
                        default='/home/ubuntu/s3-drive',
                        help="Where checkpoints from all runs are saved")
    parser.add_argument('--wandb_project', type=str, required=True,
                        help="The wandb project where all runs are tracked, of the format '<user_name>/concadia'")

    args = parser.parse_args()
    print(args)

    # Load data
    print("Loading data ...")
    with open(os.path.join(args.data_dir,'wiki_split.json'), 'r') as json_file:
        data = json.load(json_file)
    datapoints = data['images']
    data = [dp for dp in datapoints if dp['split'] == 'test']
    if args.debug:
        data = data[:2]
    print("Num testing examples: " + str(len(data)))

    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    api = wandb.Api()
    runs = api.runs(args.wandb_project)
    runs_by_data_location = {}
    for run in runs:
        if run.config['nlg_type'] not in {'resnet-lstm', 'densenet-lstm'}: continue
        run_id = run.name
        run_dir = os.path.join(args.runs_dir, 'runs', run_id)

        with open(os.path.join(run_dir, 'specs.json'), 'r') as f:
            specs = json.load(f)
        data_location = specs['data_folder'].replace("../../../../../..","")
        if data_location not in runs_by_data_location: runs_by_data_location[data_location] = []
        runs_by_data_location[data_location].append((run_id, run.config['nlg_type'], run.config['context_encoder_type']))

    for data_location, curr_runs in runs_by_data_location.items():
        # Load word map (word2ix)
        print("Loading word map")
        with open(os.path.join(data_location, "WORDMAP_wikipedia_1_min_word_freq.json"), 'r') as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

        for run_id, nlg_type, context_encoder_type in curr_runs:
            run_dir = os.path.join(args.runs_dir, 'runs', run_id)

            with open(os.path.join(run_dir, 'specs.json'), 'r') as f:
                specs = json.load(f)
            blank_context = specs['blank_context']
            if not isinstance(blank_context, bool):
                blank_context = (blank_context.lower() == 'true')

            checkpoint_dir = os.path.join(run_dir, 'checkpoints')
            for fp in sorted(os.listdir(checkpoint_dir)):
                if fp.startswith('BEST'):
                    checkpoint_path = os.path.join(checkpoint_dir, fp)

            # Sample output
            print(f"Sampling output for checkpoint {checkpoint_path} ...")

            sample_output(run_id, data, checkpoint_path, specs['label_cond'], specs['context_cond'], nlg_type, context_encoder_type, blank_context, specs['randomized'], beam_size=args.beam_size)