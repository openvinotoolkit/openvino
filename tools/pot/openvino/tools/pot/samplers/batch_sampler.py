# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.samplers.sampler import Sampler


class BatchSampler(Sampler):

    def __init__(self, data_loader, batch_size=1, subset_indices=None):
        super().__init__(data_loader, batch_size, subset_indices)
        if self._subset_indices is None:
            self._subset_indices = range(self.num_samples)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        batch = []
        for idx in self._subset_indices:
            batch.append(self._data_loader[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
