# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from addict import Dict
from ..utils.registry import Registry

ACTIVATIONS = 'activations'
WEIGHTS = 'weights'

PERCHANNEL = 'perchannel'
PERTENSOR = 'pertensor'

AGGREGATION_FN = Registry('AggregationFunctions')

ACTIVATIONS_STATS_FN = Dict(
    {'compute_statistic': {
        PERCHANNEL: Registry('ComputeActivationsPerchannelFunctions'),
        PERTENSOR: Registry('ComputeActivationsPertensorFunctions')}},
    {'statistic_in_graph': {
        PERCHANNEL: Registry('GetActivationsPerchannelFunctions'),
        PERTENSOR: Registry('GetActivationsPertensorFunctions')}})

WEIGHTS_STATS_FN = Dict({
    PERCHANNEL: Registry('WeightsPerchannelFunctions'),
    PERTENSOR: Registry('WeightsPertensorFunctions')})


def get_aggregation_function(name):
    return AGGREGATION_FN.get(name)


def get_stats_function_for_activations(name, granularity, inplace_statistics):
    return ACTIVATIONS_STATS_FN[inplace_statistics][granularity].get(name)


def get_stats_function_for_weights(name, granularity):
    return WEIGHTS_STATS_FN[granularity].get(name)


def get_stats_function(tensor_type, name, granularity, inplace_statistics):
    if tensor_type == ACTIVATIONS:
        return get_stats_function_for_activations(name, granularity, inplace_statistics)
    if tensor_type == WEIGHTS:
        return get_stats_function_for_weights(name, granularity)
    raise RuntimeError('Type of tensor is not supported. Please use {} or {} types'.format(ACTIVATIONS, WEIGHTS))
