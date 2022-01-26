# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from .functions import activations as asf


class Statistic:
    def __init__(self, func, *argv, **kwargs):
        self.func = func
        self.argv = argv
        self.kwargs = kwargs

    def compute(self, *input_tensor, **kwargs):
        pass

    def __eq__(self, other):
        if isinstance(other, Statistic):
            return self.func == other.func and self.argv == other.argv \
                   and self.kwargs == other.kwargs
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, *argv, **kwargs):
        return self.compute(*argv, **kwargs)

    def __hash__(self):
        data = (self.func, frozenset(self.argv), frozenset(self.kwargs))
        if isinstance(self.func, partial):
            data = (*data, frozenset(self.func.keywords))
        return hash(data)


class TensorStatistic(Statistic):
    def compute(self, *input_tensor, **kwargs):
        return self.func(*(input_tensor + self.argv), **self.kwargs)


# pylint: disable=W1113
class TensorStatisticAxis(Statistic):
    def __init__(self, func=None, *argv, **kwargs):
        self.func = func
        if kwargs.get('inplace_statistics', False):
            self.func = asf.get_mean_per_channel_axis
        else:
            self.func = asf.mean_per_channel_axis
        super().__init__(self.func, *argv, **kwargs)

    def compute(self, *input_tensor, **kwargs):
        return self.func(*(input_tensor + self.argv), **self.kwargs)


class SQNRStatistic(Statistic):
    def __init__(self, func, qsuffix, *argv, **kwargs):
        super().__init__(func, *argv, **kwargs)
        self.qsuffix = qsuffix

    # pylint: disable=W0221
    def compute(self, activation_dict, layer_key, **kwargs):
        return self.func(
            activation_dict[layer_key],
            activation_dict[layer_key + self.qsuffix],
            **kwargs
        )


def compute_statistic(statistic, *argv, **kwargs):
    if isinstance(argv[0], dict):
        activations_dict, layer_key = argv
        tensor = (activations_dict[layer_key],)
        if isinstance(statistic, SQNRStatistic):
            return statistic.compute(activations_dict, layer_key)
        if isinstance(statistic, TensorStatisticAxis):
            return statistic.compute(*tensor, layer_key)
    else:
        tensor = argv
    return statistic(*tensor, **kwargs)
