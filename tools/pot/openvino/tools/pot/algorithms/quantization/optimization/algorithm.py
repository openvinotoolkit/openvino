# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import warnings
import itertools
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import scipy.optimize

from ..qnoise_estimator.algorithm import QuantNoiseEstimator
from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...quantization import fake_quantize as fqut
from ....graph.model_utils import save_model
from ....samplers.creator import create_sampler
from ....utils.logger import get_logger


logger = get_logger(__name__)


try:
    import nevergrad as ng

    NEVERGRAD_AVAILABLE = True
except ImportError:
    NEVERGRAD_AVAILABLE = False
    logger.warning(
        'Nevergrad package could not be imported. If you are planning to use '
        'any hyperparameter optimization algo, consider installing it '
        'using pip. This implies advanced usage of the tool. '
        'Note that nevergrad is compatible only with Python 3.8+'
    )

try:
    import skopt

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


@COMPRESSION_ALGORITHMS.register('OptimizationAlgorithm')
class OptimizationAlgorithm(Algorithm):
    name = 'OptimizationAlgorithm'

    # pylint: disable=no-member

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._full_config = config
        default_algo = 'DoubleFastGADiscreteOnePlusOne'
        default_config = {
            'result_filename': None,
            'lower_boxsize': 0.08,
            'upper_boxsize': 0.02,
            'zero_boxsize': 0.05,
            'metric_name': 'accuracy@top1',
            'dump_model_prefix': None,
            'error_function': 'metric',
            'stochastic': False,
            'grid_search_space': None,
        }
        self._eps = 1e-8
        self._default_metrics_values = None
        self._default_opt_backend = 'nevergrad' if NEVERGRAD_AVAILABLE else 'scipy'
        self._opt_backend = self._config.get('opt_backend', self._default_opt_backend)

        for key in default_config:
            setattr(self, '_' + key, self._config.get(key, default_config[key]))

        self._optimizer_name = (
            default_algo
            if 'optimizer_name' not in self._config
            else self._config['optimizer_name']
        )
        if self._opt_backend != 'nevergrad':
            self._optimizer_name = self._opt_backend
        self._maxiter = (
            100 if 'maxiter' not in self._config else self._config['maxiter']
        )

        self._lowest_error_rate = None
        self._results = {self._metric_name: list()}
        self._optimization_iter = -1

        if not self._stochastic:
            self._subset_indices = (
                range(self._config.optimization_subset_size)
                if self._config.optimization_subset_size
                else None
            )
        else:
            self._subset_indices = (
                np.random.choice(
                    len(self._engine.data_loader), self._config.optimization_subset_size
                )
                if self._config.optimization_subset_size
                else None
            )

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        if isinstance(self._optimizer_name, str):
            self._optimizer_name = [self._optimizer_name]
            self._maxiter = [self._maxiter]
        for optimizer_name, maxiter in zip(self._optimizer_name, self._maxiter):
            logger.info(
                'Running {} optimizer for {} steps'.format(optimizer_name, maxiter)
            )
            model = self.run_single_optimization(model, maxiter, optimizer_name)
        logger.debug(
            'Lowest error value: {}'.format(self._lowest_error_rate['error'])
        )
        logger.info(
            'Lowest error parameters: {}'.format(self._lowest_error_rate['parameters'])
        )

        return model

    def run_single_optimization(self, model, maxiter, optimizer_name):
        init_parameter_values = self._get_initial_parameter_values(model)

        self._optimization_iter = -1
        # define the optimization objective function:
        self.optimization_engine(
            partial(self.evaluate_error_rate, model),
            init_parameter_values,
            maxiter,
            optimizer_name,
        )
        self._set_parameter_values(model, self._lowest_error_rate['parameters'])
        model = self._run_algo_with_params(model)
        fqut.unify_fq_scales(model, self._config)
        return model

    def evaluate_error_rate(self, model, *parameters):
        """ this function evaluates the top@1 error rate for the given parameter values
        :param model: provided model instance
        :param parameters: an array of values for parameters to set in the model
        :return evaluated top@1 error rate
        """
        self._optimization_iter += 1
        param_values = self._unpack_parameter_vector(parameters)
        self._set_parameter_values(model, param_values)
        model = self._run_algo_with_params(model)
        self._engine.set_model(model)
        error_rate, metrics = self.calculate_error_on_subset(self._subset_indices, model)

        if (
                self._lowest_error_rate is None
                or error_rate < self._lowest_error_rate['error']
        ):
            self._lowest_error_rate = {'error': error_rate, 'parameters': param_values}
        if self._default_metrics_values is None:
            self._default_metrics_values = {name: value + self._eps for name, value in metrics.items() if
                                            name != self._metric_name}
        self._results[self._metric_name].append(metrics[self._metric_name])
        if self._result_filename:
            Path('/'.join(self._result_filename.split('/')[:-1])).mkdir(
                parents=True, exist_ok=True
            )
            np.savetxt(self._result_filename, self._results, delimiter=",", fmt='%s')
        if self._dump_model_prefix:
            dump_path = self._dump_model_prefix + '{:05}'.format(
                self._optimization_iter
            )
            dump_path = Path(dump_path)
            dump_dir = dump_path.absolute().parent.as_posix()
            dump_name = dump_path.name
            save_model(model, dump_dir, dump_name)
            logger.info(
                'Model for iter {:05} is saved to {}'.format(
                    self._optimization_iter, dump_path
                )
            )
        logger.debug('All metrics: {}'.format(metrics))
        logger.debug(
            'Metric "{}" for chosen parameter vector: {}'.format(self._metric_name,
                                                                 metrics[self._metric_name])
        )
        logger.debug('Chosen parameter vector: {}'.format(param_values))
        return error_rate

    def optimization_engine(
            self, error_function, initial_guess, maxiter, optimizer_name
    ):
        """ this function performs hyperparameter value optimization
                 :param error_function: objective function to minimize taking an
                 array of parameter values
                 :param initial_guess: an array of initial guess values for parameters
                 :return minimal value of error_function reached by the optimizer
        """
        optimizer_engines = {
            'grid_search': self._gridsearch_engine,
            'nevergrad': self._nevergrad_engine,
            'scipy': self._scipy_engine,
            'skopt': self._skopt_engine,
        }
        optimizer_engines[self._opt_backend](
            error_function, initial_guess, maxiter, optimizer_name
        )

    # pylint: disable=W0613
    def _gridsearch_engine(self, error_function, initial_guess, *args):
        parameter_error_pairs = [(initial_guess, error_function(initial_guess))]
        if self._grid_search_space is not None:
            parameter_combinations = list(
                itertools.product(*self._grid_search_space.values())
            )
            for parameters_guess in parameter_combinations:
                error_value = error_function(parameters_guess)
                parameter_error_pairs.append((parameters_guess, error_value))
        return parameter_error_pairs

    def _nevergrad_engine(self, error_function, initial_guess, maxiter, optimizer_name):
        # define parameter hyperbox
        setinterval_functor = lambda range_tuple: ng.p.Array(
            init=0.5 * (range_tuple[0] + range_tuple[1])
        ).set_bounds(lower=range_tuple[0], upper=range_tuple[1])
        parameter_box = self.define_parameter_hyperbox(
            initial_guess,
            self._lower_boxsize,
            self._upper_boxsize,
            self._zero_boxsize,
            setinterval_functor=setinterval_functor,
        )
        # run optimization procedure
        instrumentation = ng.p.Instrumentation(*parameter_box)
        settings = {'parametrization': instrumentation, 'budget': maxiter}
        ng_module = importlib.import_module('nevergrad')
        optimizer_lib = getattr(ng_module, 'optimizers')
        optimizer_ = getattr(optimizer_lib, optimizer_name)
        optimizer = optimizer_(**settings)
        # estimate the initial guess point first via ask-tell interface
        parameter_error_pairs = self._gridsearch_engine(error_function, initial_guess)
        try:
            for parameter_guess, error_value in parameter_error_pairs:
                recommendation = instrumentation.spawn_child().set_standardized_data(
                    parameter_guess, deterministic=True
                )
                optimizer.tell(recommendation, error_value)
        except ng.optimization.base.TellNotAskedNotSupportedError:
            pass
        for _ in range(optimizer.budget):
            suggested_args = optimizer.ask()
            value = error_function(suggested_args.args)
            optimizer.tell(suggested_args, value)

    # pylint: disable=W0613
    def _scipy_engine(self, error_function, initial_guess, maxiter, optimizer_name):
        # define parameter hyperbox
        parameter_box = self.define_parameter_hyperbox(
            initial_guess, self._lower_boxsize, self._upper_boxsize, self._zero_boxsize
        )
        # run optimization procedure
        _ = scipy.optimize.differential_evolution(
            func=error_function,
            bounds=parameter_box,
            strategy='best2exp',
            maxiter=maxiter,
        )

    # pylint: disable=W0613
    def _skopt_engine(self, error_function, initial_guess, maxiter, optimizer_name):
        if not SKOPT_AVAILABLE:
            logger.info('Falling back on scipy opt engine')
            self._scipy_engine(error_function, initial_guess, maxiter, optimizer_name)
        # define parameter hyperbox
        parameter_box = self.define_parameter_hyperbox(
            initial_guess, self._lower_boxsize, self._upper_boxsize, self._zero_boxsize
        )
        # run optimization procedure (sequential model-based optimization)
        optimizer = skopt.Optimizer(
            dimensions=parameter_box, base_estimator='GP', n_initial_points=10
        )
        parameter_error_pairs = self._gridsearch_engine(error_function, initial_guess)
        for parameter_guess, error_value in parameter_error_pairs:
            optimizer.tell(list(parameter_guess), error_value)
        for _ in range(maxiter):
            suggested = optimizer.ask()
            error_value = error_function(suggested)
            optimizer.tell(suggested, error_value)

    @staticmethod
    def define_parameter_hyperbox(
            initial_guess,
            lower_boxsize,
            upper_boxsize,
            zero_boxsize,
            setinterval_functor=None,
    ):
        if setinterval_functor is None:
            setinterval_functor = lambda tuple: tuple
        parameter_box = []
        for value in initial_guess:
            if value > 0:
                interval = setinterval_functor(
                    ((1 - lower_boxsize) * value, (1 + upper_boxsize) * value)
                )
                parameter_box.append(interval)
            elif value < 0:
                interval = setinterval_functor(
                    ((1 + upper_boxsize) * value, (1 - lower_boxsize) * value)
                )
                parameter_box.append(interval)
            else:
                interval = setinterval_functor((-zero_boxsize, zero_boxsize))
                parameter_box.append(interval)
        return parameter_box

    def calculate_error_on_subset(self, subset_indices, model):
        def metric_error(subset_indices, model):
            self._engine.set_model(model)

            index_sampler = create_sampler(self._engine, samples=subset_indices)
            metrics, _ = self._engine.predict(None , sampler=index_sampler)

            error_rate = 0
            if self._default_metrics_values is not None:
                metrics_value = []
                for metric_name in metrics:
                    current_comparator = self._metrics_comparators[metric_name]
                    value = current_comparator(
                        (metrics[metric_name] - self._default_metrics_values[metric_name]) /
                        self._default_metrics_values[metric_name]
                    )
                    metrics_value.append(value)
                error_rate = np.negative(np.mean(metrics_value))
            metrics[self._metric_name] = error_rate
            return error_rate, metrics

        def quantization_noise(subset_indices, model):
            self._engine.set_model(model)
            estimator_config = {
                'stat_subset_size': len(subset_indices),
                'target_device': self._full_config['target_device'],
                'type': 'sqnr_eltwise_mean',
                'name': 'QuantNoiseEstimator',
            }
            noise_estimator = QuantNoiseEstimator(estimator_config, self._engine)
            noise_data = noise_estimator.full_fq_noise_stats(deepcopy(model))
            error_rate = np.sum(1 / np.array(noise_data['noise_metric']))
            return error_rate, {'quant_noise': error_rate}

        error_fn_map = {
            'metric': metric_error,
            'quantization_noise': quantization_noise,
        }
        return error_fn_map[self._error_function](subset_indices, model)

    def _get_parameter_values(self, model):
        raise NotImplementedError

    def _set_parameter_values(self, model, param_values):
        raise NotImplementedError

    def _unpack_parameter_vector(self, parameters):
        raise NotImplementedError

    def _get_initial_parameter_values(self, model):
        raise NotImplementedError
