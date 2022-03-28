# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [image_loader]
import os

import numpy as np
import cv2 as cv

from openvino.tools.pot import DataLoader

class ImageLoader(DataLoader):

    def __init__(self, dataset_path):
        """ Load images from folder  """
        # Collect names of image files
        self._files = []
        all_files_in_dir = os.listdir(dataset_path)
        for name in all_files_in_dir:
            file = os.path.join(dataset_path, name)
            if cv.haveImageReader(file):
                self._files.append(file)

        # Define shape of the model
        self._shape = (224,224)

    def __len__(self):
        """ Returns the length of the dataset """
        return len(self._files)

    def __getitem__(self, index):
        """ Returns image data by index in the NCHW layout
        Note: model-specific preprocessing is omitted, consider adding it here
        """
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        image = cv.imread(self._files[index]) # read image with OpenCV
        image = cv.resize(image, self._shape) # resize to a target input size
        image = np.expand_dims(image, 0)  # add batch dimension
        image = image.transpose(0, 3, 1, 2)  # convert to NCHW layout
        return image, None   # annotation is set to None
#! [image_loader]

#! [text_loader]
from datasets import load_dataset      #pip install datasets
from transformers import AutoTokenizer #pip install transformers

from openvino.tools.pot import DataLoader

class TextLoader(DataLoader):
    def __init__(self):
        """ HuggingFace dataset API is used to process text files """
        self.dataset = load_dataset('text', data_files='https://huggingface.co/datasets/lhoestq/test/resolve/main/some_text.txt') # replace with your text file
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')  # replace with your tokenizer
        self.dataset = self.dataset.map(self.encode, batched=True)
        self.dataset.set_format(type='numpy', columns=['input_ids', 'token_type_ids', 'attention_mask']) # replace with names of model inputs

    def encode(self, examples):
        return self.tokenizer(examples['text'], truncation=True, padding='max_length')

    def __len__(self):
        """ Returns the length of the dataset """
        return len(self.dataset)

    def __getitem__(self, index):
        """ Returns data by index as a (dict[str, np.array], None) """
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        data = self.dataset['train'][index]
        return {'input_ids': data['input_ids'], 'token_type_ids': data['token_type_ids'], 'attention_mask': data['attention_mask']}, None
#! [text_loader]