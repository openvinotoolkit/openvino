# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .data_loader import DataLoader
from .engine import Engine
from .metric import Metric
from openvino.tools.pot.utils.logger import get_logger


logger = get_logger(__name__)
logger.warning('Post-training Optimization Tool is deprecated and will be removed in the future.'
               ' Please use Neural Network Compression Framework'
               ' instead: https://github.com/openvinotoolkit/nncf')


__all__ = [
    'Metric', 'DataLoader', 'Engine'
]
