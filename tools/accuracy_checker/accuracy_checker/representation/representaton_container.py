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

import numpy as np
from ..representation import BaseRepresentation


class ContainerRepresentation(BaseRepresentation):
    def __init__(self, representation_map=None):
        super().__init__('')
        self.representations = representation_map or {}

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        if self.identifier != other.identifier:
            return False

        if self.metadata != other.metadata:
            return False

        if self.representations != other.representations:
            return False

        return True

    def __getitem__(self, item):
        return self.representations[item]

    def get(self, key):
        return self.representations.get(key)

    def values(self):
        return list(self.representations.values())

    @property
    def identifier(self):
        if self._identifier:
            return self._identifier

        values = self.values()
        if np.size(values) == 0:
            raise ValueError('representation container is empty')

        self._identifier = values[0].identifier
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier


class ContainerAnnotation(ContainerRepresentation):
    def set_image_size(self, image_sizes):
        for key in self.representations.keys():
            self.representations[key].metadata['image_size'] = image_sizes

    def set_data_source(self, data_source):
        for key in self.representations.keys():
            self.representations[key].metadata['data_source'] = data_source


class ContainerPrediction(ContainerRepresentation):
    pass
