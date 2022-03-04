# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from ..api.data_loader import DataLoader
from ..utils.ac_imports import Dataset, DatasetWrapper


class ACDataLoader(DataLoader):

    def __init__(self, data_loader: Union[Dataset, DatasetWrapper]):
        super().__init__(config=None)
        self._data_loader = data_loader

    def __len__(self):
        return self._data_loader.full_size

    def __getitem__(self, item):
        return self._data_loader[item]
