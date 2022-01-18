# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from addict import Dict
from .pipeline import Pipeline
from ..algorithms.algorithm_selector import get_algorithm
from ..utils.telemetry import send_configuration


def create_pipeline(algo_config, engine, interface='API'):
    """ Create instance of Pipeline class from config file and add specified algorithms
    :param algo_config: list of algorithms configurations
    :param engine: engine to use for inference
    :param interface: CLI or API use for inference
    :return: instance of Pipeline class
    """
    pipeline = Pipeline(engine)

    for algo in algo_config:
        if not isinstance(algo, Dict):
            algo = Dict(algo)
        algo_type = algo.name
        algo_params = algo.params
        pipeline.add_algo(get_algorithm(algo_type)(algo_params, engine))

    send_configuration(algo_config, engine, interface)
    return pipeline
