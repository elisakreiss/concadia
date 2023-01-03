import os
import numpy as np
import h5py
import json
import torch
# from scipy.misc import imread, imresize
from imageio import imread
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from PIL import Image
import pandas as pd
import glob
import sys
# import os.path
# from os import path


def create_input_files(root_dir, json_path, label, context, image_folder, labels_per_image, min_word_freq, output_folder, max_len=500):
    """
    Creates input files for training, validation, and test data.

    :param json_path: path of JSON file with splits and labels
    :param image_folder: folder with downloaded images
    :param labels_per_image: number of labels to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample labels longer than this length
    """

    dataset = "wikipedia"

    # Read JSON
    with open(os.path.join(root_dir, json_path), 'r') as j:
        data = json.load(j)

    if not os.path.exists(os.path.join(root_dir, output_folder)):
        print("creating folder at " + root_dir + output_folder)
        os.makedirs(os.path.join(root_dir, output_folder))

    files = glob.glob(os.path.join(root_dir, output_folder) + "/*")
    print(files)
    if files:
        action = input(
            '''
            There are already files created for this label and context.
            If you'd like to overwrite them, enter "y".
            If you'd like to continue without overwriting, enter "c".
            Everything else aborts process.
            Enter your choice: ''')
        if action.lower() == 'y':
            for f in files:
                os.remove(f)
            print("Removed previous files.\n")
        elif action.lower() == 'c':
            # do nothing, keep generating files
            pass
        else:
            sys.exit()

    # Read image paths and labels for each image
    train_image_paths = []
    train_image_labels = []
    train_image_contexts = []
    val_image_paths = []
    val_image_labels = []
    val_image_contexts = []
    test_image_paths = []
    test_image_labels = []
    test_image_contexts = []
    word_freq = Counter()

    for img in data['images']:
        labels = []
        contexts = []
        # Update word frequency
        # Word frequency is updated according to all text input (label + potential context)
        word_freq.update(img[label]['tokens'])
        if len(img[label]['tokens']) <= max_len:
            labels.append(img[label]['tokens'])
        if context != "none":
            contexts.append(img[context]['raw'])

        if len(labels) == 0:
            continue
        if (context != "none") and (len(contexts) == 0):
            continue

        path = os.path.join(root_dir, image_folder, 'resized', img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_labels.append(labels)
            if context != "none":
                train_image_contexts.append(contexts)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_labels.append(labels)
            if context != "none":
                val_image_contexts.append(contexts)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_labels.append(labels)
            if context != "none":
                test_image_contexts.append(contexts)

    # Sanity check
    assert len(train_image_paths) == len(train_image_labels)
    assert len(val_image_paths) == len(val_image_labels)
    assert len(test_image_paths) == len(test_image_labels)
    if context != "none":
        assert len(train_image_paths) == len(train_image_contexts)
        assert len(val_image_paths) == len(val_image_contexts)
        assert len(test_image_paths) == len(test_image_contexts)

    # Create word map
    # changed > to >= so that even unique words receive embedding
    words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(min_word_freq) + '_min_word_freq'
    print(base_filename)

    # Save word map to a JSON
    with open(os.path.join(root_dir, output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample labels for each image, save images to HDF5 file, and labels and their lengths to JSON files
    seed(123)
    for impaths, imlabs, imcontexts, split in [
        (train_image_paths, train_image_labels, train_image_contexts, 'TRAIN'),
        (val_image_paths, val_image_labels, val_image_contexts, 'VAL'),
            (test_image_paths, test_image_labels, test_image_contexts, 'TEST')]:

        with h5py.File(os.path.join(root_dir, output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of labels we are sampling per image
            h.attrs['labels_per_image'] = labels_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset(
                'images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and labels, storing to file...\n" % split)

            enc_labels = []
            lablens = []
            enc_contexts = []
            contextlens = []

            for i, path in enumerate(tqdm(impaths)):
                # print("path: ", path)

                # Sample labels
                if len(imlabs[i]) <= labels_per_image:
                    labels = imlabs[i] + [choice(imlabs[i])
                                          for _ in range(labels_per_image - len(imlabs[i]))]
                else:
                    labels = sample(imlabs[i], k=labels_per_image)

                # Sanity check
                assert len(labels) == labels_per_image

                # Read images
                if not os.path.exists(impaths[i]):
                    print("file doesn't exist: ", impaths[i])
                    continue
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = np.array(Image.fromarray(img).resize(
                    (256, 256)))  # imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(labels):
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find label lengths
                    c_len = len(c) + 2

                    enc_labels.append(enc_c)
                    lablens.append(c_len)

                if context != "none":
                    d = imcontexts[i]
                    enc_d = d

                    enc_contexts.append(enc_d)

            # Sanity check
            print(images.shape[0])
            print(labels_per_image)
            print(len(enc_labels))
            print(len(lablens))
            assert images.shape[0] * \
                labels_per_image == len(enc_labels) == len(lablens)

            # Save encoded labels and their lengths to JSON files
            with open(os.path.join(root_dir, output_folder, split + '_LABELS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_labels, j)

            with open(os.path.join(root_dir, output_folder, split + '_LABLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(lablens, j)

            if context != "none":
                with open(os.path.join(root_dir, output_folder, split + '_CONTEXTS_' + base_filename + '.json'), 'w') as j:
                    json.dump(enc_contexts, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(
            lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, cider, checkpoint_type=None):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in CIDEr score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param cider: validation CIDEr score for this epoch
    :param checkpoint_type: "best" / "last" / None
    """
    print(
        f'Saving checkpoint from epoch {epoch}, checkpoint_type={checkpoint_type}')
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'cider': cider,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = f'checkpoint_epoch{epoch:02d}.pth.tar'
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if checkpoint_type is not None:
        filename = checkpoint_type.upper() + '_' + filename
    torch.save(state, os.path.join(checkpoint_dir, filename))


def write_csv(time_dir, label_cond, context_cond, randomized, blank_img, blank_context, split, epoch, metrics_dict):

    file_path = os.path.join(time_dir, 'data.csv')

    data = {
        'label_cond': [label_cond],
        'context_cond': [context_cond],
        'randomized': [randomized],
        'blank_img': [blank_img],
        'blank_context': [blank_context],
        'split': [split],
        'epoch': [epoch],
        'avg_loss': [metrics_dict['loss']],
        'perplexity': [metrics_dict['perplexity']],
        'new_perplexity': [metrics_dict['new_perplexity']],
        'top5acc': [metrics_dict['top5acc']],
        'bleu1': [metrics_dict.get('Bleu_1', 'NA')],
        'bleu2': [metrics_dict.get('Bleu_2', 'NA')],
        'bleu3': [metrics_dict.get('Bleu_3', 'NA')],
        'bleu4': [metrics_dict.get('Bleu_4', 'NA')],
        'cider': [metrics_dict.get('CIDEr', 'NA')],
        'meteor': [metrics_dict.get('METEOR', 'NA')],
        'rouge_l': [metrics_dict.get('ROUGE_L', 'NA')],
    }
    df = pd.DataFrame(data)

    # if possible, extend existing file
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        merged_df = old_df.append(df)
        merged_df.to_csv(file_path, index=False)
    # otherwise write new file
    else:
        df.to_csv(file_path, index=False)


class Perplexity(object):
    def __init__(self):
        self._latest = 0
        self._avg = 0

    @property
    def latest(self):
        return self._latest

    @latest.setter
    def latest(self, value):
        self._latest = value

    @property
    def avg(self):
        return self._avg

    @avg.setter
    def avg(self, value):
        self._avg = value


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
