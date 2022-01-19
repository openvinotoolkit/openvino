# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools

from ..utils.logger import get_logger


logger = get_logger(__name__)


def product_dict(d):
    keys = d.keys()
    vals = d.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def check_params(algo_name, config, supported_params):
    """ Check algorithm parameters in config
        :param algo_name: name of algorithm
        :param config: config with parameters to check
        :param supported_params: parameters supported by algorithm
    """
    for key, value in config.items():
        if key not in supported_params:
            raise RuntimeError('Algorithm {}. Unknown parameter: {}'.format(algo_name, key))
        if isinstance(value, dict):
            if isinstance(supported_params[key], dict):
                check_params(algo_name, value, supported_params[key])
            else:
                raise RuntimeError('Algorithm {}. Wrong structure for parameter: {}'.format(algo_name, key))
