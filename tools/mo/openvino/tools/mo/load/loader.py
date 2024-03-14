# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils import class_registration


class Loader(object):
    registered_cls = []
    registered_ops = {}
    excluded_replacers = []

    def find_and_replace_pattern(self, graph: Graph):
        self.load(graph)

    def load(self, graph: Graph):
        raise Exception("Define load logic of {} class in its load method".format(
            self.__class__.__name__
        ))

    def run_before(self):
        """
        Returns list of loader classes which this loader must be run before.
        :return: list of classes
        """
        return [LoadFinish]

    def run_after(self):
        """
        Returns list of loader classes which this loader must be run after.
        :return: list of classes
        """
        return []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.LOADER


class LoadFinish(Loader):
    enabled = True

    def run_before(self):
        return []

    def run_after(self):
        return []

    def load(self, graph: Graph):
        graph.check_empty_graph('loading from framework')
