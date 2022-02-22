# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.engines.ac_engine import ACEngine
from openvino.tools.pot.engines.simplified_engine import SimplifiedEngine


def create_engine(config, **kwargs):
    """
    Factory to create instance of engine class based on config
    :param config: engine config section from toolkit config file
    :param kwargs: additional arguments specific for every engine class (data_loader, metric)
    :return: instance of Engine descendant class
    """
    if config.type == 'accuracy_checker':
        return ACEngine(config)
    if config.type == 'simplified':
        return SimplifiedEngine(config, **kwargs)
    raise RuntimeError('Unsupported engine type')
