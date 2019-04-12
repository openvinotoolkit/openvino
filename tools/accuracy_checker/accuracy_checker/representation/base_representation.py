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

import abc
import pickle


class BaseRepresentation(abc.ABC):
    def __init__(self, identifier, metadata=None):
        self.identifier = identifier
        self.metadata = metadata or {}

    @classmethod
    def load(cls, file):
        obj = pickle.load(file)

        if cls != BaseRepresentation:
            assert isinstance(obj, cls)

        return obj

    def dump(self, file):
        pickle.dump(self, file)

    def set_image_size(self, image_sizes):
        self.metadata['image_size'] = image_sizes

    def set_data_source(self, data_source):
        self.metadata['data_source'] = data_source
