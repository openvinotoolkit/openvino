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

from ...utils import read_pickle
from .loader import Loader, DictLoaderMixin


class PickleLoader(DictLoaderMixin, Loader):
    """
    Class for loading output from another tool in .pickle format.
    """

    __provider__ = 'pickle'

    def load(self):
        data = read_pickle(self._data_path)

        if isinstance(data, list) and all(hasattr(entry, 'identifier') for entry in data):
            return dict(zip([representation.identifier for representation in data], data))

        return data
