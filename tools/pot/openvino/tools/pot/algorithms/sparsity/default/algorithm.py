# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base_algorithm import BaseWeightSparsity
from ....algorithms.algorithm import Algorithm
from ....algorithms.algorithm_selector import COMPRESSION_ALGORITHMS
from ....utils.logger import get_logger

# pylint: disable=W0611
try:
    import torch
    from ..layerwise_finetuning.algorithm import SparseModelFinetuning

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('WeightSparsity')
class WeightSparsity(Algorithm):
    name = 'WeightSparsity'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        use_layerwise_tuning = self._config.get('use_layerwise_tuning', False)
        if use_layerwise_tuning:
            if TORCH_AVAILABLE:
                self.algo_pipeline = SparseModelFinetuning(config, engine)
            else:
                raise ModuleNotFoundError('Cannot import the torch package which is a dependency '
                                          'of the LayerwiseFinetuning algorithm. '
                                          'Please install torch via `pip install torch==1.6.0')
        else:
            self.algo_pipeline = BaseWeightSparsity(config, engine)

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        self.algo_pipeline.run(model)
        logger.info('\n' + self.algo_pipeline.statistics_table.draw())
        return model
