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

from ...utils import read_xml
from .loader import Loader, DictLoaderMixin


class XMLLoader(DictLoaderMixin, Loader):
    """
    Class for loading output from another tool in .xml format.
    """

    __provider__ = 'xml'

    def load(self):
        return read_xml(self._data_path)
