"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import re

from ..config import PathField
from ..representation import ClassificationAnnotation
from ..utils import get_path, read_txt

from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


class SampleConverterConfig(BaseFormatConverterConfig):
    data_dir = PathField(is_directory=True)


class SampleConverter(BaseFormatConverter):
    """
    Sample dataset converter. All annotation converters should be derived from BaseFormatConverter class.
    """

    # register name for this converter
    # this name will be used for converter class look up
    __provider__ = 'sample'

    _config_validator_type = SampleConverterConfig

    def configure(self):
        self.data_dir = self.config['data_dir']

    def convert(self):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically forwarded from command line arguments.

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata.
        """

        dataset_directory = get_path(self.data_dir, is_directory=True)

        # read and convert annotation
        labels = self._read_labels(dataset_directory / 'labels.txt')
        annotations = self._convert_annotations(dataset_directory / 'test', labels)

        # convert label list to label map
        label_map = {i: labels[i] for i in range(len(labels))}
        metadata = {'label_map': label_map}

        return annotations, metadata

    @staticmethod
    def _read_labels(labels_file):
        """
        Extract label names from labels.txt file.
        """

        return read_txt(labels_file)

    @staticmethod
    def _convert_annotations(test_dir, labels):
        """
        Create annotation representations list.
        """

        # test directory contains files with names XXXX_class.png
        # we use regular expression to extract class names
        file_pattern_regex = re.compile(r'\d+_(\w+)\.png')

        annotations = []
        # iterate over all png images in test directory
        for image in test_dir.glob('*.png'):
            # get file name (e.g. from /foo/bar/image.png we get image.png)
            image_base = str(image.parts[-1])

            # extract class name from file name
            regex_match = re.match(file_pattern_regex, image_base)
            image_label = regex_match.group(1)

            # look up class index in label list
            class_id = labels.index(image_label)

            # create annotation representation object
            annotations.append(ClassificationAnnotation(image_base, class_id))

        return annotations
