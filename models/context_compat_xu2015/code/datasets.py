import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import h5py
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class LabelDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, context_cond="none", blank_img=False, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # labels per image
        self.cpi = self.h.attrs['labels_per_image']

        self.context_cond = context_cond
        self.blank_img = blank_img

        # Load encoded labels (completely into memory)
        with open(os.path.join(data_folder, self.split + '_LABELS_' + data_name + '.json'), 'r') as j:
            self.labels = json.load(j)

        # Load label lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_LABLENS_' + data_name + '.json'), 'r') as j:
            self.lablens = json.load(j)
        
        if self.context_cond != "none":
            # Load encoded contexts (completely into memory)
            with open(os.path.join(data_folder, self.split + '_CONTEXTS_' + data_name + '.json'), 'r') as j:
                self.contexts = json.load(j)


        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.labels)

    def __getitem__(self, i):
        # Remember, the Nth label corresponds to the (N // labels_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        
        if self.blank_img:
            img = torch.ones(img.size())

        label = torch.LongTensor(self.labels[i])
        lablen = torch.LongTensor([self.lablens[i]])

        if self.context_cond != "none":
            context = self.contexts[i][0]
        else:
            context = []

        if self.split is 'TRAIN':
            return img, label, lablen, context
        else:
            # For validation of testing, also return all 'labels_per_image' labels to find BLEU-4 score
            all_labels = torch.LongTensor(
                self.labels[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, label, lablen, all_labels, context

    def __len__(self):
        return self.dataset_size
