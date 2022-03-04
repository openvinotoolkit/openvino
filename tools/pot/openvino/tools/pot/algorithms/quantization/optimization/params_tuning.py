# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
from copy import deepcopy
from .algorithm import OptimizationAlgorithm
from ..accuracy_aware_common.utils import create_metric_config
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('ParamsGridSearchAlgorithm')
class ParamsGridSearchAlgorithm(OptimizationAlgorithm):
    name = 'ParamsGridSearchAlgorithm'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._grid_search_space = [
            {
                # range_estimator params
                'weights': {
                    'min': [
                        {'type': 'min'}
                    ],
                    'max': [
                        {'type': 'max'}
                    ]
                },
                'activations': {
                    'min': [
                        {'type': 'min', 'aggregator': 'min'},
                        {'type': 'quantile', 'aggregator': 'mean', 'outlier_prob': 10e-4},
                        {'type': 'quantile', 'aggregator': 'mean', 'outlier_prob': 10e-5}
                    ],
                    'max': [
                        {'type': 'max', 'aggregator': 'max'},
                        {'type': 'quantile', 'aggregator': 'mean', 'outlier_prob': 10e-4},
                        {'type': 'quantile', 'aggregator': 'mean', 'outlier_prob': 10e-5}
                    ]
                }
            },
            {
                # bias correction variations
                'use_fast_bias': [True, False]
            }
        ]
        self._range_estimator_types = ['activations', 'weights']
        self._opt_backend = 'grid_search'
        self._metric_name = 'avg_metric_err'
        self._results = {self._metric_name: list()}
        self._optimizer_name = self.name
        self._metrics_comparators = create_metric_config(self._engine, self._config)
        self._lowest_error_rate = None
        base_algo = self._config.get('base_algorithm', 'DefaultQuantization')

        self._quantization_algo = COMPRESSION_ALGORITHMS.get(base_algo)(
            config, engine
        )
        dataset_size = len(self._engine.data_loader)
        self._stat_subset_size = min(self._config.get('stat_subset_size', dataset_size), dataset_size)
        self._subset_indices = range(self._stat_subset_size)
        self._default_config = deepcopy(self._quantization_algo.config)

    @property
    def default_algo(self):
        return self._quantization_algo

    @default_algo.setter
    def default_algo(self, algo):
        self._quantization_algo = algo
        self._default_config = deepcopy(self._quantization_algo.config)

    def set_subset_and_metric(self, subset, metrics_params):
        '''
        This method overrides subset and params if need
        '''
        self._subset_indices = subset
        self._metrics_comparators = {metric_name: metrics_params[metric_name]['comparator'] for metric_name in
                                     metrics_params}

    def _gridsearch_engine(self, error_function, initial_guess, *args):
        '''
        This method uses gridsearch-like engine to create and run new params combinations
        '''
        self._quantization_algo.update_config(self._default_config)
        parameter_error_pairs = []

        # run quantization algo with combinations
        parameter_combinations = self._create_parameter_combinations()
        for space in parameter_combinations:
            additional_parameters = self._lowest_error_rate['parameters'] if self._lowest_error_rate is not None else []
            for parameters_guess in space:
                parameters_guess.extend(additional_parameters)
                error_value = error_function(parameters_guess)
                parameter_error_pairs.append((parameters_guess, error_value))

        return parameter_error_pairs

    def _create_parameter_combinations(self):
        '''
        This method creates combinations using _grid_search_space values
        '''
        parameter_combinations = []
        for search_space in self._grid_search_space:
            parameter_combination = []
            for space_name, space_values in search_space.items():
                if space_name in self._range_estimator_types:
                    combinations = self._create_special_combination(space_name, space_values)
                else:
                    combinations = self._create_usual_combination(space_name, [space_values])
                parameter_combination.extend(combinations)
            parameter_combinations.append(parameter_combination)
        return parameter_combinations

    def _create_special_combination(self, space_name, space_values):
        '''
        This method created combination for 'weights' and 'activation' search spaces
        '''
        prepared_values = []
        for range_estimator_type, space_value in space_values.items():
            prepared_values.append([[range_estimator_type, value] for value in space_value])
        return self._create_usual_combination(space_name, prepared_values)

    def _create_modified_config(self, params_data):
        '''
        This method creates new config with given parameters
        '''
        modified_config = deepcopy(self._default_config)

        if params_data:
            for param_data in params_data:
                params_namespace = param_data[0]
                param_values = param_data[1]

                if params_namespace in self._range_estimator_types:
                    modified_config[params_namespace] = {
                        'range_estimator': {}
                    }
                    for estimator_type, values in param_values:
                        for param_name, param_value in values.items():
                            estimator_config = modified_config[params_namespace]['range_estimator']
                            if estimator_type not in estimator_config:
                                estimator_config[estimator_type] = {}
                            estimator_config[estimator_type].update({param_name: param_value})
                else:
                    modified_config[params_namespace] = param_values[0]
        return modified_config

    def _set_parameter_values(self, model, param_values):
        '''
        This method sets new parametes values
        and updates quantization algo configutation
        '''
        algo_config = self._create_modified_config(param_values)
        self._quantization_algo.update_config(algo_config)

    def _get_initial_parameter_values(self, model):
        if self._lowest_error_rate is None:
            init_parameter_values = []
        else:
            init_parameter_values = self._lowest_error_rate['parameters']
        return init_parameter_values

    def _run_algo_with_params(self, model):
        return self._quantization_algo.run(deepcopy(model))

    def _unpack_parameter_vector(self, parameters):
        return parameters[0]

    def _get_parameter_values(self, model):
        pass

    def register_statistics(self, model, stats_collector):
        self._stats_collector = stats_collector

        # register statistics for combinations
        parameter_combinations = self._create_parameter_combinations()
        self._maxiter = sum(len(space) for space in parameter_combinations)
        for space in parameter_combinations:
            for parameters in space:
                logger.debug('Register stats for next options: {}'.format(parameters))
                self._set_parameter_values(model, parameters)
                self._quantization_algo.register_statistics(model, stats_collector)
            self._quantization_algo.update_config(self._default_config)

    def update_config(self, config):
        self._default_config = deepcopy(config)

    @staticmethod
    def _create_usual_combination(space_name, space_values):
        return [[(space_name, tup)] for tup in itertools.product(*space_values)]
