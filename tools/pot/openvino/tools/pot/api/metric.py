# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class Metric(ABC):
    """An abstract class representing an accuracy metric. """

    def __init__(self):
        """ Constructor """
        self.reset()

    @property
    def value(self):
        """ Returns accuracy metric value for the last model output. """
        raise Exception('The value() property should be implemented to use this metric '
                        'with AccuracyAwareQuantization algorithm!')

    @property
    @abstractmethod
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """

    @abstractmethod
    def update(self, output, target):
        """ Calculates and updates accuracy metric value
        :param output: model output
        :param target: annotations
        """

    @abstractmethod
    def reset(self):
        """ Reset accuracy metric """

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    @property
    def higher_better(self):
        """Attribute whether the metric should be increased"""
        return True

    @abstractmethod
    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
