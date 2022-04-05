# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .algorithms.quantization.accuracy_aware.algorithm import AccuracyAwareQuantization
from .algorithms.quantization.accuracy_aware_gna.algorithm import AccuracyAwareGNA
from .algorithms.quantization.accuracy_aware_common.algorithm import AccuracyAwareCommon
from .algorithms.quantization.accuracy_aware_common.mixed_precision import (
    INT4MixedQuantization,
)
from .algorithms.quantization.fast_bias_correction.algorithm import FastBiasCorrection
from .algorithms.quantization.bias_correction.algorithm import BiasCorrection
from .algorithms.quantization.channel_alignment.algorithm import (
    ActivationChannelAlignment,
)
from .algorithms.quantization.datafree.algorithm import DataFreeQuantization
from .algorithms.quantization.default.algorithm import DefaultQuantization
from .algorithms.quantization.minmax.algorithm import MinMaxQuantization
from .algorithms.quantization.optimization.rangeopt import RangeOptimization
from .algorithms.quantization.optimization.params_tuning import (
    ParamsGridSearchAlgorithm,
)
from .algorithms.quantization.qnoise_estimator.algorithm import QuantNoiseEstimator
from .algorithms.quantization.tunable_quantization.algorithm import TunableQuantization
from .algorithms.quantization.outlier_channel_splitting.algorithm import (
    OutlierChannelSplitting,
)
from .algorithms.quantization.weight_bias_correction.algorithm import (
    WeightBiasCorrection,
)
from .algorithms.sparsity.magnitude_sparsity.algorithm import MagnitudeSparsity
from .algorithms.sparsity.default.algorithm import WeightSparsity
from .algorithms.sparsity.default.base_algorithm import BaseWeightSparsity
from .algorithms.quantization.overflow_correction.algorithm import OverflowCorrection
from .algorithms.quantization.ranger.algorithm import Ranger

from .api.data_loader import DataLoader
from .api.metric import Metric
from .api.engine import Engine
from .engines.ie_engine import IEEngine
from .graph import load_model, save_model
from .graph.model_utils import compress_model_weights
from .pipeline.initializer import create_pipeline

QUANTIZATION_ALGORITHMS = [
    'MinMaxQuantization',
    'RangeOptimization',
    'FastBiasCorrection',
    'BiasCorrection',
    'ActivationChannelAlignment',
    'DataFreeQuantization',
    'DefaultQuantization',
    'AccuracyAwareQuantization',
    'AccuracyAwareGNA',
    'AccuracyAwareCommon',
    'INT4MixedQuantization',
    'TunableQuantization',
    'QuantNoiseEstimator',
    'OutlierChannelSplitting',
    'WeightBiasCorrection',
    'ParamsGridSearchAlgorithm',
    'OverflowCorrection',
    'Ranger',
]

SPARSITY_ALGORITHMS = ['WeightSparsity',
                       'MagnitudeSparsity',
                       'BaseWeightSparsity']

__all__ = QUANTIZATION_ALGORITHMS + SPARSITY_ALGORITHMS
