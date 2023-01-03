import time
import argparse
import copy
import os
from datetime import datetime
import sys
import json
import csv

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from modeling import Encoder, DecoderWithAttention, DecoderWithContext, DecoderWithContextRevised
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from sentence_transformers import SentenceTransformer
from nlgeval import NLGEval
import wandb

NUM_BATCHES_IN_DEBUG = 2

# print("loading models")
nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'EmbeddingAverageCosineSimilarity',
                                   'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity',
                                   'GreedyMatchingScore'])  # loads the models

# SET LABEL AND CONTEXT
blank_img = False
blank_context_zeros = False
context_model = "revised"  # original, revised
max_context_len = 52  # maxlen in dataset

context_encoders = {
    'none': 'bert-base-uncased',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base'
}

image_encoder_dims = {
    'resnet': 2048,
    'densenet': 2208,
}

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
# sets device for model and PyTorch tensors
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('torch.cuda.is_available(): ', torch.cuda.is_available())
print('device: ', device)
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True
loss_function = nn.CrossEntropyLoss()

# Training parameters
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?

save_checkpoint_frequency = 5


def run_training():
    """
    Training and validation.
    """
    global word_map, rev_word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    # Initialize / load checkpoint
    if args.blank_context:
        context_encoder_type = 'none'
    else:
        context_encoder_type = args.context_encoder_type
    context_encoder_path = context_encoders[context_encoder_type]
    context_tokenizer = AutoTokenizer.from_pretrained(context_encoder_path)
    context_encoder_path = context_encoders[context_encoder_type]
    start_epoch = 0
    best_epoch = 0
    best_cider = 0  # best CIDEr score so far
    # keeps track of number of epochs since there's been an improvement in validation BLEU
    epochs_since_improvement = 0

    decoder = DecoderWithContextRevised(attention_dim=attention_dim,
                                        embed_dim=emb_dim,
                                        decoder_dim=decoder_dim,
                                        vocab_size=len(word_map),
                                        dropout=dropout,
                                        encoder_dim=image_encoder_dims[args.image_encoder_type],
                                        context_encoder_path=context_encoder_path)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=decoder_lr)
    encoder = Encoder(encoder_type=args.image_encoder_type)
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=encoder_lr) if fine_tune_encoder else None

    nlg_type = f'{args.image_encoder_type}-lstm'
    wandb.init(project="concadia",
               job_type='train',
               dir=run_dir,
               name=folder_name,
               config={**specs_dict, 'run_dir': run_dir, 'nlg_type': nlg_type,
                       'context_encoder_type': context_encoder_type},
               group=f"label={args.label_cond}, context={'none' if args.blank_context else args.context_cond}, randomized={args.randomized}",
               tags=[nlg_type, context_encoder_type])

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = loss_function.to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        LabelDataset(data_folder, data_name, 'TRAIN', context_cond=args.context_cond, blank_img=blank_img,
                     transform=transforms.Compose([normalize])), batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        LabelDataset(data_folder, data_name, 'VAL', context_cond=args.context_cond, blank_img=blank_img,
                     transform=transforms.Compose([normalize])), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train_metrics = train(train_loader=train_loader,
                              encoder=encoder,
                              decoder=decoder,
                              criterion=criterion,
                              encoder_optimizer=encoder_optimizer,
                              decoder_optimizer=decoder_optimizer,
                              epoch=epoch,
                              context_tokenizer=context_tokenizer)

        # One epoch's validation
        # recent_bleu4, rec_loss, rec_t5acc = validate(val_loader=val_loader,
        val_metrics, val_metrics_greedy_decoding = validate(val_loader=val_loader,
                                                            encoder=encoder,
                                                            decoder=decoder,
                                                            criterion=criterion,
                                                            epoch=epoch,
                                                            context_tokenizer=context_tokenizer)

        recent_cider = val_metrics_greedy_decoding["CIDEr"]

        # Check if there was an improvement
        is_best = recent_cider > best_cider
        if is_best:
            best_cider = recent_cider
            best_epoch = epoch
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)
            best_encoder_optimizer = copy.deepcopy(encoder_optimizer)
            best_decoder_optimizer = copy.deepcopy(decoder_optimizer)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))

        # Save checkpoint regularly and save last checkpoint
        if epoch % save_checkpoint_frequency == 0 or epoch == args.epochs - 1:
            checkpoint_type = "last" if epoch == args.epochs - 1 else None
            save_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, encoder, decoder,
                            encoder_optimizer, decoder_optimizer, recent_cider, checkpoint_type=checkpoint_type)

        write_csv(run_dir, args.label_cond, args.context_cond, args.randomized,
                  blank_img, args.blank_context, "val", epoch, val_metrics)

        wandb.log(format_log_metrics(train_metrics, val_metrics,
                  val_metrics_greedy_decoding, epoch))

    if start_epoch > best_epoch:
        print('Current resumed run does not have a new best checkpoint.')
    else:
        wandb.run.summary['best_epoch'] = best_epoch
        wandb.run.summary['best_val_cider'] = best_cider
        # Save best checkpoint
        save_checkpoint(checkpoint_dir, best_epoch, 0, best_encoder, best_decoder,
                        best_encoder_optimizer, best_decoder_optimizer, best_cider, checkpoint_type='best')


def update_metrics(losses: AverageMeter,
                   nlls: AverageMeter,
                   perplexities: AverageMeter,
                   perplexity: Perplexity,
                   top5accs: AverageMeter,
                   loss: float,
                   nll: float,
                   top5acc,
                   num_tokens: int):
    losses.update(loss, num_tokens)
    nlls.update(nll, num_tokens)
    perplexities.update(np.exp(loss), num_tokens)
    top5accs.update(top5acc, num_tokens)

    curr_batch_perplexity = np.exp(nlls.val)
    avg_perplexity = np.exp(nlls.avg)
    perplexity.latest = curr_batch_perplexity
    perplexity.avg = avg_perplexity


def format_log_metrics(train_m, val_m, val_from_scratch_m, epoch):
    metrics = {'epoch': epoch}
    metrics.update({f'{key}/train': val for key, val in train_m.items()})
    metrics.update({f'{key}/val': val for key, val in val_m.items()})
    metrics.update({f'{key}/val_from_scratch': val for key,
                   val in val_from_scratch_m.items()})
    return metrics


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, context_tokenizer):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    torch.cuda.empty_cache()
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    nlls = AverageMeter()
    perplexities = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    new_perplexity = Perplexity()

    start = time.time()

    # Batches
    for i, (imgs, labs, lablens, contexts) in enumerate(train_loader):
        if args.debug and i == NUM_BATCHES_IN_DEBUG:
            break
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        labs = labs.to(device)
        lablens = lablens.to(device)
        if not args.context_cond == "none":
            tokenized_contexts = context_tokenizer(
                contexts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_context_len)
            contexts = tokenized_contexts.input_ids.to(device)
            context_mask = tokenized_contexts.attention_mask.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        if args.context_cond == "none":
            scores, labs_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, labs, lablens)
        else:
            scores, labs_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, labs, lablens, contexts, context_mask, args.blank_context, blank_context_zeros)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = labs_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, * \
            _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, * \
            _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        raw_loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss = raw_loss + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        update_metrics(losses, nlls, perplexities, new_perplexity, top5accs,
                       loss.item(), raw_loss.item(), top5, sum(decode_lengths))

        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #   'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Perplex {perplexity.val:.4f} ({perplexity.avg:.4f})\t'
                  'NEW Perplex {new_perplexity.latest:.4f} ({new_perplexity.avg:.4f})\t'
                  'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                     batch_time=batch_time,
                                                                     data_time=data_time, loss=losses,
                                                                     perplexity=perplexities,
                                                                     new_perplexity=new_perplexity,
                                                                     top5=top5accs))
    metrics_to_log = {
        'loss': losses.avg,
        'perplexity': perplexities.avg,
        'new_perplexity': new_perplexity.avg,
        'top5acc': top5accs.avg
    }

    write_csv(run_dir, args.label_cond, args.context_cond, args.randomized,
              blank_img, args.blank_context, "train", epoch, metrics_to_log)

    return metrics_to_log


def validate(val_loader, encoder, decoder, criterion, epoch, context_tokenizer):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    nlls = AverageMeter()
    perplexities = AverageMeter()
    top5accs = AverageMeter()
    new_perplexity = Perplexity()

    references = list()  # references (true labels) for calculating BLEU-4 score
    ref_words = list()
    hypotheses = list()  # hypotheses (predictions)
    hyp_words = list()
    hypotheses_from_scratch = list()
    hyp_words_from_scratch = list()

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, labs, lablens, alllabs, contexts) in enumerate(val_loader):
            if args.debug and i == NUM_BATCHES_IN_DEBUG:
                break

            start = time.time()
            # Move to device, if available
            imgs = imgs.to(device)
            labs = labs.to(device)
            lablens = lablens.to(device)
            # COMMENT OUT
            if not args.context_cond == "none":
                tokenized_contexts = context_tokenizer(
                    contexts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_context_len)
                contexts = tokenized_contexts.input_ids.to(device)
                context_mask = tokenized_contexts.attention_mask.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, labs_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, labs, lablens, contexts, context_mask, args.blank_context, blank_context_zeros)
            bsz = labs.shape[0]
            seqs_from_scratch = torch.LongTensor(
                [[word_map['<start>']]] * bsz).to(device)
            scores_from_scratch, _, _, _, _ = decoder(imgs, seqs_from_scratch, lablens, contexts, context_mask,
                                                        args.blank_context, blank_context_zeros, greedy_decode_from_scratch=True)

            # ===== Calculate ML metrics =====
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = labs_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.detach()
            scores, * \
                _ = pack_padded_sequence(
                    scores, decode_lengths, batch_first=True)
            targets, * \
                _ = pack_padded_sequence(
                    targets, decode_lengths, batch_first=True)

            # Calculate loss
            raw_loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss = raw_loss + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            update_metrics(losses, nlls, perplexities, new_perplexity, top5accs,
                           loss.item(), raw_loss.item(), top5, sum(decode_lengths))

            batch_time.update(time.time() - start)

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Perplex {perplexity.val:.4f} ({perplexity.avg:.4f})\t'
                      'NEW Perplex {new_perplexity.latest:.4f} ({new_perplexity.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                perplexity=perplexities,
                                                                                new_perplexity=new_perplexity,
                                                                                loss=losses, top5=top5accs))

            # ===== Calculate NLG metrics =====
            # Store references (true labels), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # because images were sorted in the decoder
            alllabs = alllabs[sort_ind]
            for j in range(alllabs.shape[0]):
                img_labs = alllabs[j].tolist()
                img_labels = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>'], word_map['<end>']}],
                        img_labs))  # remove <start> and pads
                references.append(img_labels)
                ref_words.append(
                    " ".join([rev_word_map[ind] for ind in img_labels[0]]))

            # Hypotheses
            def convert_preds_to_hyps(ps, pred_lengths=None):
                ps = ps.tolist()
                hs = list()
                h_words = list()
                for j, p in enumerate(ps):
                    if pred_lengths is not None:
                        hs.append(p[:pred_lengths[j]])  # remove pads
                    else:
                        # find where the sequence ends
                        # if no <end> token found, keep the entire seq
                        end_idx = len(p)
                        for idx, token in enumerate(p):
                            if token == word_map['<end>']:
                                end_idx = idx + 1
                                break
                        hs.append(p[:end_idx])
                for h in hs:
                    h_words.append(" ".join(
                        [rev_word_map[ind] for ind in h if rev_word_map[ind] not in {'<end>', '<pad>'}]))
                return hs, h_words

            _, preds = torch.max(scores_copy, dim=2)
            curr_hyps, curr_hyp_words = convert_preds_to_hyps(preds)
            hypotheses.extend(curr_hyps)
            hyp_words.extend(curr_hyp_words)
            assert len(references) == len(hypotheses)

            # Hypotheses from scratch
            preds_from_scratch = scores_from_scratch.argmax(dim=2)
            curr_hyps_from_scratch, curr_hyp_words_from_scratch = convert_preds_to_hyps(
                preds_from_scratch)
            hypotheses_from_scratch.extend(curr_hyps_from_scratch)
            hyp_words_from_scratch.extend(curr_hyp_words_from_scratch)
            assert len(references) == len(hypotheses_from_scratch)

        r = [ref_words]
        h = hyp_words
        h_from_scratch = hyp_words_from_scratch

        # This metric still includes the <end> token in both which explains why the metrics aren't 0
        metrics_dict = nlgeval.compute_metrics(r, h)
        metrics_dict_from_scratch = nlgeval.compute_metrics(r, h_from_scratch)

        # write val results to json
        metric_debug_json = []
        for ref, hyp, hyp_from_scratch in zip(ref_words, hyp_words, hyp_words_from_scratch):
            datapoint = {
                'reference': ref,
                'hypothesis': {'text': hyp},
                'hypothesis_greedy': {'text': hyp_from_scratch}
            }
            for hypothesis_type in ['hypothesis', 'hypothesis_greedy']:
                metrics = nlgeval.compute_metrics(
                    [[ref]], [datapoint[hypothesis_type]['text']])
                for metric in ['Bleu_2', 'Bleu_4', 'ROUGE_L']:
                    # not recording CIDEr since it's based on corpus statistics
                    datapoint[hypothesis_type][metric] = metrics[metric]
            metric_debug_json.append(datapoint)
        filename = os.path.join(run_dir, f'valdata_epoch{epoch:02d}.json')
        with open(filename, 'w') as json_file:
            json.dump(metric_debug_json, json_file, indent=4)


        def format_nlg_metrics(m):
            return f"BLEU-2 - {m['Bleu_2']:.5f}, BLEU-4 - {m['Bleu_4']:.5f}, CIDEr - {m['CIDEr']:.5f}, ROUGE_L - {m['ROUGE_L']:.3f}"

        print(
            f'\n * LOSS - {losses.avg:.3f}, PERPLEX - {perplexities.avg:.3f}, NEW PERPLEX - {new_perplexity.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}\n'
            f'\t GT DECODING: {format_nlg_metrics(metrics_dict)}\n'
            f'\t DECODING FROM SCRATCH: {format_nlg_metrics(metrics_dict_from_scratch)}\n')

    additional_metrics_to_log = {
        'loss': losses.avg,
        'perplexity': perplexities.avg,
        'new_perplexity': new_perplexity.avg,
        'top5acc': top5accs.avg
    }

    metrics_dict.update(additional_metrics_to_log)

    # return bleu4, losses, top5accs
    return metrics_dict, metrics_dict_from_scratch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training configs.')
    parser.add_argument('label_cond', type=str, choices=['description', 'caption'],
                        help='Output type.')
    parser.add_argument('context_cond', type=str, choices=['description', 'caption', 'context'],
                        help='Context type (what to input aside from img).')
    parser.add_argument('randomized', type=str, choices=['img', 'description', 'caption', 'context', 'none'],
                        help='Type of input to randomize.')
    parser.add_argument('--image_encoder_type', type=str, choices=['resnet', 'densenet'], default='resnet',
                        help='Image encoder type.')
    parser.add_argument('--context_encoder_type', type=str, choices=['bert', 'roberta'], default='bert',
                        help='Context encoder type.')
    parser.add_argument('--blank_context', action='store_true',
                        help='Whether to replace context with a constant mask.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Turn on debug mode.')
    parser.add_argument('--epochs', type=int, default=35,
                        help='Number of epochs.')
    parser.add_argument('--data_dir', type=str,
                        default='/home/ubuntu/s3-drive/data/concadia_xu2015_data',
                        help="Where data for model training and eval is stored")
    parser.add_argument('--output_dir', type=str,
                        default='/home/ubuntu/s3-drive/',
                        help="Where to output run metrics and checkpoints")

    args = parser.parse_args()

    # Data parameters
    data_folder = os.path.join(args.data_dir, args.label_cond + '_' + args.context_cond +
                               '_' + 'random' + args.randomized)  # folder with data files saved by create_input_files.py
    data_name = 'wikipedia_1_min_word_freq'  # base name shared by data files

    if args.debug:
        # dryrun
        args.epochs = 2
        print_freq = 1

    # create folder for current run
    now = datetime.now()
    folder_name = now.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, 'runs', folder_name)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')

    os.makedirs(checkpoint_dir)  # make both run dir and checkpoint dir
    print('\n' + f'Storing all files in {folder_name}' + '\n')

    # save specs in current run folder
    specs_dict = {
        'data_name': data_name,
        'data_folder': data_folder,
        'label_cond': args.label_cond,
        'context_cond': args.context_cond,
        'randomized': args.randomized,
        'blank_img': blank_img,
        'blank_context': args.blank_context,
        'blank_context_zeros': blank_context_zeros,
        'context_model': context_model,
        'emb_dim': str(emb_dim),
        'attention_dim': str(attention_dim),
        'decoder_dim': str(decoder_dim),
        'dropout': str(dropout),
        'device': str(device),
        'cudnn.benchmark': str(cudnn.benchmark),
        'loss_function': str(loss_function),
        'batch_size': str(batch_size),
        'workers': str(workers),
        'encoder_lr': str(encoder_lr),
        'decoder_lr': str(decoder_lr),
        'grad_clip': str(grad_clip),
        'alpha_c': str(alpha_c),
        'fine_tune_encoder': str(fine_tune_encoder),
    }

    with open(os.path.join(run_dir, 'specs.json'), 'w') as json_file:
        json.dump(specs_dict, json_file, indent=4)

    print("Starting initialization")

    run_training()
