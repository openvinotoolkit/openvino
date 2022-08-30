# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random

from openvino.tools.pot.engines.ac_engine import ACEngine
from openvino.tools.pot.samplers.batch_sampler import BatchSampler
from openvino.tools.pot.samplers.index_sampler import IndexSampler


def create_sampler(engine, samples, shuffle_data=False, seed=0, batch_size=1):
    """ Helper function to create the most common samplers. Suits for the most algorithms
    :param engine: instance of engine class
    :param samples: a list of dataset indices or a number of samples to draw from dataset
    :param shuffle_data: a boolean flag. If it's True and samples param is a number then
     subset indices will be choice randomly
    :param seed: a number for initialization of the random number generator
    :param batch_size: a number for batch_size for IEEngine
    :return instance of Sampler class suitable to passed engine
    """

    if isinstance(samples, int):
        if shuffle_data:
            random.seed(seed)
            samples = random.sample(range(len(engine.data_loader)), samples)
        else:
            samples = range(samples)

    if isinstance(engine, ACEngine):
        return IndexSampler(subset_indices=samples)

    return BatchSampler(engine.data_loader, batch_size=batch_size, subset_indices=samples)
