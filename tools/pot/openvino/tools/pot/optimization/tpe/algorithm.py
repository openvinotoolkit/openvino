# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

# pylint: disable=import-error
import numpy as np
import pandas as pd
import hyperopt as hpo
from hyperopt import fmin, hp, STATUS_OK, Trials

from openvino.tools.pot.benchmark.benchmark import benchmark_embedded, set_benchmark_config
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.optimization.optimizer import Optimizer
from openvino.tools.pot.optimization.tpe.multinode import Multinode
from openvino.tools.pot.samplers.index_sampler import IndexSampler
from openvino.tools.pot.utils.logger import get_logger
from openvino.tools.pot.utils.object_dump import object_load, object_dump

logger = get_logger(__name__)


class TpeOptimizer(Optimizer):
    def __init__(self, config, pipeline, engine):
        super().__init__(config, pipeline, engine)
        self._search_space = None
        self._hpopt_trials = None
        self._hpopt_search_space = None
        self._evaluated_params = []
        self._tpe_params = {
            'n_initial_point': 10,
            'gamma': 0.3,
            'n_EI_candidates': 100,
            'prior_weight': 1.0
        }
        # Experimental option, currently not exposed to customer in external documentation
        if self._config.get('anneal', False):
            self._algo = hpo.anneal.suggest
        else:
            self._algo = None

        self._fp32_acc = None
        self._fp32_lat = None
        self._engine.calculate_metrics = True
        self._base_metrics_list, _ = zip(*self._engine.get_metrics_attributes().items())
        self._opt_param = None
        self._best_result = {
            'best_loss': float('inf'),
            'best_acc_loss': float('inf'),
            'best_lat_diff': 0.0,
            'best_quantization_ratio': 0.0
        }

        self.multinode = None
        self._start_time = None
        self._trial_load = self._config.get('trials_load_method', 'cold_start')
        self._debug = self._config.get('debug', False)

        self._loss_function_config = self._prepare_loss_function_config(self._config)
        if 'benchmark' in self._config:
            set_benchmark_config(self._config.benchmark)

    def _prepare_loss_function_config(self, config):
        return {
            'acc_th': config.get('accuracy_loss', 1) / 100,
            'lat_th': config.get('latency_reduce', 1),
            'quant_ratio_offset': config.get('expected_quantization_ratio', 0.5),
            'acc_weight': config.get('accuracy_weight', 1.0),
            'lat_weight': config.get('latency_weight', 1.0),
            'quant_ratio_weight': config.get('quantization_ratio_weight', 0.0)
        }

    def _create_search_space(self, model, optimizer_state):
        param_grid = []
        for algo in self._pipeline.algo_seq:
            for param in algo.get_parameter_meta(model, optimizer_state):
                # Make param name unique among algorithms
                unique_param_name = algo.name + '_' + param[0]
                param_grid.append((unique_param_name, param[1], param[2]))
        return param_grid

    def _configure_hpopt_search_space_and_params(self, search_space):
        self._hpopt_search_space = {}
        for param in search_space:
            if param[1] == 'choice':
                self._hpopt_search_space[param[0]] = hp.choice(param[0], param[2])
            else:
                raise ValueError('Unsupported param type: {}'.format(param[1]))
        # Find minimum number of choices for params with more than one choice
        multichoice_params = [len(param[2]) for param in search_space if param[1] == 'choice' and len(param[2]) > 1]
        min_param_size = min(multichoice_params) if multichoice_params else 1
        if not self._config.get('anneal', False):
            self._tpe_params['n_EI_candidates'] = min_param_size
            self._tpe_params['prior_weight'] = 1 / min_param_size
            self._algo = partial(hpo.tpe.suggest,
                                 n_startup_jobs=self._tpe_params['n_initial_point'],
                                 gamma=self._tpe_params['gamma'],
                                 n_EI_candidates=self._tpe_params['n_EI_candidates'],
                                 prior_weight=self._tpe_params['prior_weight'])

    def generate_quantized_model(self, params, model):
        self._set_algorithms_params(params)
        return self._pipeline.run(deepcopy(model))

    def _set_algorithms_params(self, params):
        for algo in self._pipeline.algo_seq:
            algo_params = {}
            for unique_param_name in [param for param in params.keys() if param.startswith(algo.name)]:
                param_name = unique_param_name[len(algo.name) + 1:]
                algo_params[param_name] = deepcopy(params[unique_param_name])
            algo.params = algo_params

    def object_evaluation(self, params, model):
        if params in [ep['params'] for ep in self._evaluated_params]:
            previous_result = [ep for ep in self._evaluated_params if ep['params'] == params][0]
            previous_result['reevaluate_count'] += 1
            logger.info(
                'Params already evaluated. Returning previous result: acc_loss: {:.4f} lat_diff: {:.4f}'.format(
                    previous_result['result']['acc_loss'], previous_result['result']['lat_diff']))
            return previous_result['result']
        if self._debug:
            object_dump(params, os.path.join(self._config.exec_log_dir, 'tpe_' + model.name + '_last_trial_params.p'))
        result = self._compute_metrics_for_params(params, model)
        self._update_result_with_calculated_loss(result)
        self._evaluated_params.append(
            {'params': params, 'result': result,
             'first_iteration': len(self._hpopt_trials.trials),
             'reevaluate_count': 0})
        return result

    def _compute_metrics_for_params(self, params, model):
        qmodel = self.generate_quantized_model(params, model)
        lat = None
        #calculate benchmark on server
        if self.multinode is not None:
            if self._hpopt_trials is not None:
                lat = self.multinode.request_remote_benchmark(qmodel, len(self._hpopt_trials.trials))
            else:
                lat = self.multinode.request_remote_benchmark(qmodel, 0)
        # get int8 accuracy metric
        logger.info('Compute INT8 metrics')
        acc_new, lat_new = self._model_eval(qmodel, lat)
        quantization_params = [param for param in params.values() if 'quantize' in param]
        if not quantization_params:
            quantization_ratio = 1.0
        else:
            quantization_ratio = len([param for param in quantization_params if param['quantize'] == 1]) \
                                 / len(quantization_params)
        quantization_params = []
        for algo in self._pipeline.algo_seq:
            for param in algo.params.values():
                if 'quantize' in param:
                    quantization_params.append(param)
        if not quantization_params:
            real_quantization_ratio = 1.0
        else:
            real_quantization_ratio = len([param for param in quantization_params if param['quantize'] == 1]) \
                                      / len(quantization_params)
        fp32_acc = self._fp32_acc[self._base_metrics_list[0]]
        int8_acc = acc_new[self._base_metrics_list[0]]
        acc_diff = (fp32_acc - int8_acc) / fp32_acc
        lat_diff = self._fp32_lat / lat_new
        return {
            'acc': acc_new,
            'lat': lat_new,
            'acc_loss': acc_diff,
            'lat_diff': lat_diff,
            'quantization_ratio': quantization_ratio,
            'real_quantization_ratio': real_quantization_ratio,
            'exec_time': (time.time() - self._start_time) / 60,
            'params': params,
            'status': STATUS_OK,
            'node': self.multinode.name if self.multinode is not None else "no_name"}

    def _update_result_with_calculated_loss(self, result):
        result['loss'] = self.calculate_loss(result['acc_loss'], result['lat_diff'], result['real_quantization_ratio'],
                                             self._loss_function_config)
        logger.info('Current iteration acc_loss: {:.4f} lat_diff: {:.4f} exec_time: {:.4f}'.format(
            result['acc_loss'], result['lat_diff'], result['exec_time']))

    @staticmethod
    def _calculate_acc_loss_component(acc_loss):
        return np.exp(acc_loss)

    @staticmethod
    def _calculate_lat_diff_component(lat_diff):
        return np.log(np.power((1 / (1000 * lat_diff)), 8))

    @staticmethod
    def _calculate_loss_function_scaling_components(acc_loss, lat_diff, config):
        acc_min = TpeOptimizer._calculate_acc_loss_component(0)
        acc_max = TpeOptimizer._calculate_acc_loss_component(acc_loss)
        if acc_max == acc_min:
            acc_max = TpeOptimizer._calculate_acc_loss_component(config['acc_th'])
        config['acc_min'] = acc_min
        config['acc_scale'] = 10 / np.abs(acc_max - acc_min)

        lat_min = TpeOptimizer._calculate_lat_diff_component(lat_diff)
        lat_max = TpeOptimizer._calculate_lat_diff_component(1)
        if lat_min == lat_max:
            lat_min = TpeOptimizer._calculate_lat_diff_component(config['lat_th'])
        config['lat_min'] = lat_min
        config['lat_scale'] = 10 / np.abs(lat_max - lat_min)

    @staticmethod
    def calculate_loss(acc_diff, lat_diff, quantization_ratio, config):
        gamma_penalty = 40  # penalty term
        quant_ratio_scaling_factor = 1.0 / min(1.0 - config['quant_ratio_offset'], config['quant_ratio_offset'])
        acc_loss_component = TpeOptimizer._calculate_acc_loss_component(acc_diff)
        lat_loss_component = TpeOptimizer._calculate_lat_diff_component(lat_diff)
        acc_weight = config['acc_weight'] if acc_diff > config['acc_th'] else 0.0
        if acc_weight == 0 and config['lat_weight'] == 0 and config['quant_ratio_weight'] == 0:
            acc_weight = 1.0
        loss = acc_weight * (config['acc_scale'] * (acc_loss_component - config['acc_min'])) \
               + config['lat_weight'] * (config['lat_scale'] * (lat_loss_component - config['lat_min'])) \
               - config['quant_ratio_weight'] * 5 \
               * np.tanh(3 * quant_ratio_scaling_factor * (quantization_ratio - config['quant_ratio_offset']))
        if acc_diff > config['acc_th']:
            loss += 2 * gamma_penalty
        elif lat_diff < config['lat_th']:
            loss += gamma_penalty
        return loss

    def _model_eval(self, model, lat):
        if not self._config.keep_uncompressed_weights:
            compress_model_weights(model)
        self._engine.set_model(model)
        subset_indices = range(self._config.eval_subset_size) \
            if self._config.eval_subset_size else range(len(self._engine.data_loader))
        acc, _ = self._engine.predict(sampler=IndexSampler(subset_indices=subset_indices),
                                      print_progress=self._debug)
        if lat is None:
            lat = benchmark_embedded(model)
        return acc, lat

    def _update_best_result(self, best_result_file):
        if not self._hpopt_trials:
            raise Exception(
                'No trials loaded to get best result')
        trials_results = pd.DataFrame(self._hpopt_trials.results)

        if not trials_results[trials_results.acc_loss <= self._loss_function_config['acc_th']].empty:
            # If accuracy threshold reached, choose best latency
            best_result = trials_results[trials_results.acc_loss <= self._loss_function_config['acc_th']] \
                .reset_index(drop=True).sort_values(by=['lat_diff', 'acc_loss'], ascending=[False, True]) \
                .reset_index(drop=True).loc[0]
        else:
            # If accuracy threshold is not reached, choose based on loss function
            best_result = trials_results.sort_values('loss', ascending=True).reset_index(drop=True).loc[0]

        update_best_result = False
        if not self._best_result['best_loss']:
            update_best_result = True
        elif self._best_result['best_acc_loss'] <= self._loss_function_config['acc_th']:
            if best_result['acc_loss'] <= self._loss_function_config['acc_th'] \
                    and best_result['lat_diff'] > self._best_result['best_lat_diff']:
                update_best_result = True
        else:
            if best_result['acc_loss'] <= self._loss_function_config['acc_th'] or \
                    best_result['loss'] < self._best_result['best_loss']:
                update_best_result = True

        if update_best_result:
            object_dump(dict(best_result), best_result_file)
            best_result.to_csv(best_result_file + '.csv', header=False)
            self._opt_param = best_result['params']
            self._best_result['best_loss'] = best_result['loss']
            self._best_result['best_acc_loss'] = best_result['acc_loss']
            self._best_result['best_lat_diff'] = best_result['lat_diff']
            self._best_result['best_quantization_ratio'] = best_result['real_quantization_ratio']

        logger.info('Trial iteration end: {} / {} acc_loss: {:.4f} lat_diff: {:.4f} '.format(
            len(self._hpopt_trials.trials), self._config.max_trials, self._best_result['best_acc_loss'],
            self._best_result['best_lat_diff']))

    def _restore_best_result(self, best_result_file):
        """ restore trial database and state"""
        best_result = object_load(best_result_file)
        self._opt_param = best_result['params']
        self._best_result['best_loss'] = best_result['loss']
        self._best_result['best_acc_loss'] = best_result['acc_loss']
        self._best_result['best_lat_diff'] = best_result['lat_diff']
        self._best_result['best_quantization_ratio'] = best_result['real_quantization_ratio']

    def _create_tuned_loss_config(self, loss_config_file, model):
        search_space = self._create_search_space(model, {'first_iteration': False, 'fully_quantized': True})
        params = {}
        for param in search_space:
            if param[1] == 'choice':
                params[param[0]] = param[2][0]
            else:
                raise ValueError('Unsupported param type: {}'.format(param[1]))
        result = self._compute_metrics_for_params(params, model)
        TpeOptimizer._calculate_loss_function_scaling_components(result['acc_loss'], result['lat_diff'],
                                                                 self._loss_function_config)
        self._update_result_with_calculated_loss(result)
        self._evaluated_params.append(
            {'params': params, 'result': result,
             'first_iteration': 0,
             'reevaluate_count': 0})
        self._set_algorithms_params({})
        object_dump(self._loss_function_config, loss_config_file)

    def _restore_tuned_loss_config(self, loss_config_file):
        """ restore loss function configuration"""
        loss_config = object_load(loss_config_file)
        self._loss_function_config['acc_min'] = loss_config['acc_min']
        self._loss_function_config['acc_scale'] = loss_config['acc_scale']
        self._loss_function_config['lat_min'] = loss_config['lat_min']
        self._loss_function_config['lat_scale'] = loss_config['lat_scale']

    def _run_trials(self, max_trials, max_minutes, trials_file, best_result_file, model):
        trials_count = len(self._hpopt_trials.trials)
        while trials_count < max_trials:
            if max_minutes != 0 and (time.time() - self._start_time) > (max_minutes * 60):
                logger.info('Time limit of {} minutes reached'.format(max_minutes))
                break
            trials_count += 1
            logger.info('Trial iteration start: {} / {}'.format(trials_count, max_trials))
            fmin(partial(self.object_evaluation, model=model),
                 space=self._hpopt_search_space,
                 algo=self._algo,
                 max_evals=trials_count,
                 trials=self._hpopt_trials,
                 show_progressbar=False)
            self._save_trials(trials_file)
            self._update_remote_trials()
            trials_count = len(self._hpopt_trials.trials)
            self._update_best_result(best_result_file)
            if self._config.get('stop_on_target', False) \
                    and self._best_result['best_acc_loss'] <= self._loss_function_config['acc_th'] \
                    and self._best_result['best_lat_diff'] >= self._loss_function_config['lat_th']:
                logger.info('Accuracy and latency reached')
                break

    def _get_fp32_metrics(self, model, fp32_metric_file):
        # get fp32 metrics from .p file
        if Path(fp32_metric_file).exists():
            logger.info('load fp32 metrics directly from {}'.format(fp32_metric_file))
            self._fp32_acc, self._fp32_lat = object_load(fp32_metric_file)
        else:
            # get metrics from config .json file if exist
            self._fp32_acc = {metric.name: metric.baseline_value
                              for metric in self._config.metrics
                              if metric.name in self._base_metrics_list} \
                if self._config.metrics and \
                   all('baseline_value' in metric.keys() for metric in self._config.metrics) \
                else None
            if self._fp32_acc is not None:
                logger.info('load fp32_acc metric directly from config file')
                # if fp32_acc alredy loaded from config run benchmark only for fp32_lat
                logger.info('load fp32_lat metric from benchmark')
                self._fp32_lat = benchmark_embedded(model)
            # if can't find metrics in config file nor in .p file run evaluation
            else:
                logger.info('compute fp32 metrics once and save to {}'.format(fp32_metric_file))
            self._fp32_acc, self._fp32_lat = self._model_eval(model, None)
            object_dump((self._fp32_acc, self._fp32_lat), fp32_metric_file)
            pd.Series({'acc': self._fp32_acc, 'lat': self._fp32_lat}).to_csv(fp32_metric_file + '.csv',
                                                                             header=False)
            if self.multinode is not None:
                self._fp32_lat = self.multinode.update_or_restore_fp32(self._fp32_acc, self._fp32_lat)

    def _get_tuned_loss_config(self, model, loss_config_file):
        if Path(loss_config_file).exists():
            logger.info('load loss function config directly from {}'.format(loss_config_file))
            self._restore_tuned_loss_config(loss_config_file)
        else:
            logger.info('compute loss function config and save in {}'.format(loss_config_file))
            self._create_tuned_loss_config(loss_config_file, model)

    def _update_remote_trials(self):
        if self.multinode is not None:
            self._hpopt_trials = self.multinode.update_remote_trials(self._hpopt_trials)
            self._evaluated_params = self.multinode.update_remote_evaluated_params(self._evaluated_params)

    def _restore_remote_trials(self):
        self._hpopt_trials = self.multinode.restore_remote_trials()
        self._evaluated_params = self.multinode.restore_remote_evaluated_params()

    def _update_remote_startup_data(self):
        if self.multinode is not None:
            self.multinode.update_remote_search_space(self._search_space)
            self._update_remote_trials()

    def _compute_first_iteration_and_create_final_search_space(self, model, tpe_config_file, best_result_file,
                                                               trials_file):
        self._hpopt_trials = Trials()
        # Run first iteration
        self._search_space = self._create_search_space(model,
                                                       {'first_iteration': True, 'fully_quantized': False})
        self._configure_hpopt_search_space_and_params(self._search_space)
        logger.info('Trial iteration start: {} / {}'.format(1, self._config.max_trials))
        fmin(partial(self.object_evaluation, model=model),
             space=self._hpopt_search_space,
             algo=self._algo,
             max_evals=1,
             trials=self._hpopt_trials,
             show_progressbar=False)
        self._update_best_result(best_result_file)

        # Generate final search space and TPE config
        self._search_space = self._create_search_space(model,
                                                       {'first_iteration': False, 'fully_quantized': False})
        self._configure_hpopt_search_space_and_params(self._search_space)
        self._update_remote_startup_data()
        # Remove trials file before updating config and trials to prevent mismatch between config and trials files
        if Path(trials_file).exists():
            os.remove(trials_file)
        self._save_tpe_config(tpe_config_file)
        self._save_trials(trials_file)

    def start_trials(self, model, fp32_metric_file, tpe_config_file, trials_file, best_result_file, loss_config_file):
        self._start_time = time.time()
        self._get_fp32_metrics(model, fp32_metric_file)
        self._get_tuned_loss_config(model, loss_config_file)

        if self.multinode is not None and self.multinode.type == 'client':
            self._search_space = self.multinode.restore_remote_search_space()
            self._configure_hpopt_search_space_and_params(self._search_space)
            self._restore_remote_trials()

        elif self._trial_load == 'warm_start':
            self._restore_tpe_config(tpe_config_file)
            self._restore_trials(trials_file)
            # If trials were restored upload this data to remote database
            self._update_remote_startup_data()
            logger.info('Rerunning from {} trials to {} trials'
                        .format(len(self._hpopt_trials.trials), self._config.max_trials))

        elif self._trial_load == 'cold_start':
            self._compute_first_iteration_and_create_final_search_space(model, tpe_config_file, best_result_file,
                                                                        trials_file)

        elif self._trial_load == 'fine_tune':
            self._restore_best_result(best_result_file)
            logger.info('Found best result! acc_loss: {:.4f} lat_diff: {:.4f} '.format(
                self._best_result['best_acc_loss'], self._best_result['best_lat_diff']))
            self._set_algorithms_params(self._opt_param)
            # Remove best loss to "reset" best_loss in _update_best_result
            self._best_result['best_loss'] = None
            self._compute_first_iteration_and_create_final_search_space(model, tpe_config_file, best_result_file,
                                                                        trials_file)

        if self.multinode is not None and self.multinode.type == 'server' and self.multinode.mode == 'master':
            self.multinode.calculate_remote_requests()
            self._restore_remote_trials()
            self._update_best_result(best_result_file)
        else:
            self._run_trials(self._config.max_trials,
                             self._config.max_minutes if self._config.max_minutes else 0,
                             trials_file, best_result_file, model)

        if self._best_result['best_acc_loss'] <= self._loss_function_config['acc_th'] \
                and self._best_result['best_lat_diff'] >= self._loss_function_config['lat_th']:
            logger.info('Congratulation parameters found to meet latency and accuracy criteria')
        elif self._best_result['best_acc_loss'] <= self._loss_function_config['acc_th']:
            logger.info('TPE satisfy accuracy criteria but can not satisfy latency criteria')
        else:
            logger.info('Sorry, TPE can not satisfy accuracy and latency criteria')

    def _save_tpe_config(self, tpe_config_file):
        tpe_config_object = self._search_space
        object_dump(tpe_config_object, tpe_config_file)

    def _restore_tpe_config(self, tpe_config_file):
        self._search_space = object_load(tpe_config_file)
        self._configure_hpopt_search_space_and_params(self._search_space)

    def _save_trials(self, trials_log):
        """ save trial result to log file"""
        trials_object = (self._hpopt_trials, self._evaluated_params)
        object_dump(trials_object, trials_log)
        tpe_trials_results = pd.DataFrame(self._hpopt_trials.results)
        csv_file = trials_log + '.csv'
        tpe_trials_results.to_csv(csv_file)
        tpe_trials_results = pd.DataFrame(self._evaluated_params)
        csv_file = trials_log + '.params.csv'
        tpe_trials_results.to_csv(csv_file)

    def _restore_trials(self, trials_log):
        """ restore trial database and state"""
        trials_object = object_load(trials_log)

        self._hpopt_trials = trials_object[0]
        self._evaluated_params = trials_object[1]

    def run(self, model):
        """ this function applies TPE algorithm
         :param model: model to apply algo
         :return model with inserted and filled FakeQuantize nodes
         """
        if self._config.eval_subset_size:
            fp32_metric_file = 'tpe_' + model.name + '_fp32_metric_' + str(self._config.eval_subset_size) + '.p'
        else:
            fp32_metric_file = 'tpe_' + model.name + '_fp32_metric.p'
        fp32_metric_file = os.path.join(self._config.model_log_dir, fp32_metric_file)
        trials_file = os.path.join(self._config.model_log_dir, 'tpe_' + model.name + '_trials.p')
        tpe_config_file = os.path.join(self._config.model_log_dir, 'tpe_' + model.name + '_config.p')
        best_result_file = os.path.join(self._config.model_log_dir, 'tpe_' + model.name + '_best_result.p')
        loss_config_file = os.path.join(self._config.model_log_dir, 'tpe_' + model.name + '_loss_config.p')

        if 'multinode' in self._config:
            self.multinode = Multinode(self._config, model.name)
            config = self.multinode.update_or_restore_config(self._config)
            if self.multinode.type == "client":
                self._loss_function_config = self._prepare_loss_function_config(config)
                self._config.max_trials = config.get('max_trials', None)
                self._config.max_minutes = config.get('max_minutes', None)
                self._config.eval_subset_size = config.get('eval_subset_size', None)
                set_benchmark_config(config['benchmark'])

        # Remove any old data on cold_start
        if self._trial_load == 'cold_start':
            if Path(fp32_metric_file).exists():
                os.remove(fp32_metric_file)
            if Path(trials_file).exists():
                os.remove(trials_file)
            if Path(tpe_config_file).exists():
                os.remove(tpe_config_file)
            if Path(best_result_file).exists():
                os.remove(best_result_file)
            if Path(loss_config_file).exists():
                os.remove(loss_config_file)

        if self._trial_load != 'eval':
            self.start_trials(model, fp32_metric_file, tpe_config_file, trials_file, best_result_file, loss_config_file)
            if self.multinode is not None:
                self.multinode.cleanup()
        else:
            # restore best result for eval
            if Path(best_result_file).exists():
                self._restore_best_result(best_result_file)
            else:
                raise Exception(
                    'WARNING: Best result file {} does not exist '.format(best_result_file))
        logger.info(' -Generate final INT8 model now')
        final_model = self.generate_quantized_model(self._opt_param, model)

        return final_model
