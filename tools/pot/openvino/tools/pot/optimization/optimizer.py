# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from ..api.engine import Engine
from ..pipeline.pipeline import Pipeline


class Optimizer(ABC):
    def __init__(self, config, pipeline: Pipeline, engine: Engine):
        """ Constructor
         :param config: optimizer config
         :param pipeline: pipeline of algorithms to optimize
         :param engine: entity responsible for communication with dataset
          """
        self._config, self._pipeline, self._engine = config.params, pipeline, engine
        self.name = config.name

    @abstractmethod
    def run(self, model):
        """ Run optimizer on model
        :param model: model to apply optimization
         """
