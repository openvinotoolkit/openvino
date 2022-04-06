# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from addict import Dict


class DataLoader(ABC):
    """An abstract class representing a dataset.

    All custom datasets should inherit.
    ``__len__`` provides the size of the dataset and
    ``__getitem__`` supports integer indexing in range from 0 to len(self)
    """

    def __init__(self, config):
        """ Constructor
        :param config: data loader specific config
        """
        self.config = config if isinstance(config, Dict) else Dict(config)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass
