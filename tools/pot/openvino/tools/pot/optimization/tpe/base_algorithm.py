# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

try:
    # pylint: disable=unused-import
    import hyperopt
    from .algorithm import TpeOptimizer

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False


from openvino.tools.pot.optimization.optimizer import Optimizer
from openvino.tools.pot.optimization.optimizer_selector import OPTIMIZATION_ALGORITHMS


@OPTIMIZATION_ALGORITHMS.register('Tpe')
class Tpe(Optimizer):
    def __init__(self, config, pipeline, engine):
        super().__init__(config, pipeline, engine)
        if HYPEROPT_AVAILABLE:
            self.optimizer = TpeOptimizer(config, pipeline, engine)
        else:
            raise ModuleNotFoundError(
                'Cannot import the hyperopt package which is a dependency '
                'of the TPE algorithm. '
                'Please install hyperopt via `pip install hyperopt==0.1.2 pandas==0.24.2`'
            )

    def run(self, model):
        return self.optimizer.run(model)
