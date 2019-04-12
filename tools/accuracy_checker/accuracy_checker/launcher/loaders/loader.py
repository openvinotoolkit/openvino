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

from pathlib import Path

from ...dependency import ClassProvider


class Loader(ClassProvider):
    """
    Interface that describes loading output from another tool.
    """

    __provider_type__ = 'loader'

    def __init__(self, data_path: Path):
        self._data_path = data_path

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class DictLoaderMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if item not in self.data:
            raise IndexError('There is no prediction object for "{}" input data'.format(item))

        return self.data[item]

    def load(self):
        raise NotImplementedError
