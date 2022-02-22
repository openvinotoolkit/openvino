# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.samplers.sampler import Sampler
from .utils import format_input_batch


class IndexSampler(Sampler):

    def __init__(self, subset_indices):
        super().__init__(subset_indices=subset_indices)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for idx in self._subset_indices:
            yield idx

    def __getitem__(self, idx):
        data_batch = self._subset_indices[idx]
        formatted_batch = data_batch if isinstance(data_batch, dict) else format_input_batch(data_batch, idx)
        return formatted_batch
