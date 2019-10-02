"""
Copyright (C) 2018-2019 Intel Corporation

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

import os
import numpy
import pickle
import shutil
import tempfile
from typing import Dict


class InferRawResults:
    def __init__(self):
        self._size = 0
        self._index = 0
        self._dir_path = None
        pass

    def release(self):
        if self._dir_path:
            shutil.rmtree(self._dir_path)
            self._dir_path = None

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < self._size:
            file_path = os.path.join(self._dir_path, str(self._index))
            self._index += 1

            f = open(file_path, "rb")
            try:
                loaded_value = pickle.load(f)
            finally:
                f.close()
            return loaded_value
        else:
            raise StopIteration

    def size(self):
        return self._size

    def add(self, value: Dict[str, numpy.ndarray]):
        if self._dir_path is None:
            self._dir_path = tempfile.mkdtemp("__infer_raw_results")
            if not os.path.exists(self._dir_path):
                os.makedirs(self._dir_path)

        file_path = os.path.join(self._dir_path, str(self._size))

        f = open(file_path, "wb")
        try:
            pickle.dump(value, f)
        finally:
            f.close()

        self._size += 1
