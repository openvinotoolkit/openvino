# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.samplers.sampler import Sampler


class IndexSampler(Sampler):

    def __init__(self, subset_indices):
        super().__init__(subset_indices=subset_indices)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for idx in self._subset_indices:
            yield idx

    def __getitem__(self, idx):
        return self._subset_indices[idx]
