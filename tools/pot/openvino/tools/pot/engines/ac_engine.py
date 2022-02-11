# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import multiprocessing
from collections import OrderedDict
from functools import partial

import numpy as np

from .utils import append_stats, process_accumulated_stats, \
    restore_original_node_names, align_stat_names_with_results, \
    add_tensor_names, cast_friendly_names, collect_model_outputs
from ..api.engine import Engine
from ..data_loaders.ac_data_loader import ACDataLoader
from ..graph.model_utils import save_model, add_outputs
from ..utils.ac_imports import create_model_evaluator, _DEFAULT_LOGGER_NAME
from ..utils.logger import get_logger, stdout_redirect
from ..utils.utils import create_tmp_dir, convert_output_key

logger = get_logger(__name__)


# pylint: disable=R0912
class ACEngine(Engine):

    def __init__(self, config):
        """ Constructor
        :param config: accuracy checker config
         """
        super().__init__(config)
        dataset_config = config.models[0].datasets if config.models else config.evaluations[0].module_config.datasets
        self._evaluation_dataset_tag = 'evaluation' if isinstance(dataset_config, dict) else ''
        self._optimization_dataset_tag = 'optimization' if isinstance(dataset_config, dict) else ''
        self._model_evaluator = create_model_evaluator(config)
        self.select_dataset(self._evaluation_dataset_tag)
        self._model = None
        self._nx_model = None
        self._accumulated_layer_stats = dict()
        self._per_sample_metrics = list()
        self._tmp_dir = create_tmp_dir()
        self._dataset_tag = None
        self.allow_pairwise_subset = False
        self.dump_prediction_to_annotation = False
        self.calculate_metrics = False
        self.annotation_conf_threshold = 0.0

    def _set_model_from_files(self, paths):
        # disable accuracy checker info logging
        logging.getLogger(_DEFAULT_LOGGER_NAME).setLevel(logging.WARNING)
        # load IR model
        self._model = self._load_model(paths)
        # reset model evaluator metrics
        self._model_evaluator.reset()
        self._model_evaluator.load_network(self._model)
        # turn it on to print metrics
        logging.getLogger(_DEFAULT_LOGGER_NAME).setLevel(logging.INFO)

    def set_model_from_files(self, paths):
        """ Load NetworkX model into InferenceEngine from .xml and .bin files
        and stores it in Engine class
        :param paths: list of dictionaries
        'name': name of the model (only for cascaded models)
        'model': path to the .xml model file,
        'weights': path to the .bin weights file
        """
        stdout_redirect(self._set_model_from_files, paths)

    def set_model(self, model):
        """ Load NetworkX model into InferenceEngine and stores it in Engine class
        :param model: CompressedModel instance
        """
        def _set_model(path):
            tmp_model_name = 'tmp_model'
            paths = save_model(model, path, tmp_model_name, for_stat_collection=True)
            self._set_model_from_files(paths)
            self._nx_model = model

        stdout_redirect(_set_model, self._tmp_dir.name)

    def set_dataset_tag(self, dataset_tag: str):
        """ Sets the dataset tag of accuracy checker that will be used in
        the future calls of the method `predict`.
        Note that if a dataset tag is not set by the method, but
        configuration of accuracy checker contains several dataset tags, then
        dataset_tag='optimization' is used for optimization, and
        dataset_tag='evaluation' is used for evaluation
         :param dataset_tag: dataset tag that should be used
        """
        self._dataset_tag = dataset_tag
        if dataset_tag is not None:
            self.select_dataset(dataset_tag)

    def select_dataset(self, dataset_tag):
        """ Sets data_loader by a dataset tag """
        stdout_redirect(self._model_evaluator.select_dataset, dataset_tag)
        self._data_loader = ACDataLoader(self._model_evaluator.dataset)

    @property
    def dataset_tag(self):
        return self._dataset_tag

    def predict(self, stats_layout=None, sampler=None, stat_aliases=None,
                metric_per_sample=False, print_progress=False):
        """ Performs model inference on specified dataset subset
         :param stats_layout: dict of stats collection functions {node_name: {stat_name: fn}} (optional)
         :param sampler: entity to make dataset sampling
         :param stat_aliases: dict of algorithms collections stats
                {algorithm_name: {node_name}: {stat_name}: fn} (optional)
         :param metric_per_sample: if Metric is specified and the value is True,
                then the metric value will be calculated for each data sample, otherwise for the whole dataset.
         :param print_progress: whether to print inference progress
         :returns a tuple of dictionaries of persample and overall metric values if 'metric_per_sample' is True
                  ({sample_id: sample index, 'metric_name': metric name, 'result': metric value},
                   {metric_name: metric value}), otherwise, a dictionary of overall metrics
                   {metric_name: metric value}
                  a dictionary of collected statistics {node_name: {stat_name: [statistics]}}
        """
        if self._model is None:
            raise Exception('Model wasn\'t set in Engine class')

        logger_fn = logger.info if print_progress else logger.debug

        callback_layout, stat_names_aliases = {}, {}
        # add outputs for activation statistics collection
        if stats_layout is not None:
            model_with_stat_op, nodes_names_map, output_to_node_names = self._statistic_graph_builder.\
                insert_statistic(copy.deepcopy(self._nx_model),
                                 stats_layout, stat_aliases)
            self.set_model(model_with_stat_op)

            for model in self._model:
                cast_friendly_names(model['model'].outputs)

            outputs_per_model = add_outputs(self._model, nodes_names_map)
            for model_name, outputs_data in outputs_per_model.items():
                add_tensor_names(outputs_data, nodes_names_map[model_name].keys())

            self._model_evaluator.load_network(self._model)

            model_output_names = []
            for model in self._model:
                model_output_names.extend(collect_model_outputs(model['model']))

            nodes_name = []
            for names_map in nodes_names_map.values():
                nodes_name.extend(list(names_map.keys()))

            align_stat_names_with_results(model_output_names,
                                          nodes_name,
                                          output_to_node_names,
                                          stats_layout,
                                          stat_aliases)

            # Creating statistics layout with IE-like names
            stat_names_aliases = {convert_output_key(key): key for key in stats_layout}
            callback_layout = {convert_output_key(key): value
                               for key, value in stats_layout.items()}

        if self._dataset_tag is None:
            dataset_tag = self._optimization_dataset_tag \
                if stats_layout else self._evaluation_dataset_tag
            logger.debug('Dataset tag is set to "{}"'.format(dataset_tag))
        else:
            dataset_tag = self._dataset_tag
            logger.debug('Dataset tag is set manually to "{}"'.format(dataset_tag))
        self.select_dataset(dataset_tag)

        if sampler:
            logger_fn('Start inference of {} images'.format(len(sampler)))
        else:
            logger_fn('Start inference on the whole dataset')

        args = {'subset': sampler,
                'output_callback': partial(self._ac_callback, callback_layout, metric_per_sample),
                'check_progress': print_progress,
                'dataset_tag': dataset_tag,
                'dump_prediction_to_annotation': self.dump_prediction_to_annotation,
                'calculate_metrics': self.calculate_metrics
                }
        if sampler:
            args['allow_pairwise_subset'] = self.allow_pairwise_subset
        if self.dump_prediction_to_annotation:
            args['annotation_conf_threshold'] = self.annotation_conf_threshold

        requests_number = self._stat_requests_number if stats_layout else self._eval_requests_number
        if requests_number == 1:
            stdout_redirect(self._model_evaluator.process_dataset, **args)
        else:
            self._set_requests_number(args, requests_number)
            stdout_redirect(self._model_evaluator.process_dataset_async, **args)

        logger_fn('Inference finished')
        if self.calculate_metrics:
            metrics = OrderedDict([
                (
                    metric.name, np.mean(metric.evaluated_value)
                    if metric.meta.get('calculate_mean', True) else metric.evaluated_value[0]
                )
                for metric in stdout_redirect(self._model_evaluator.compute_metrics, print_results=False)
            ])
        else:
            metrics = OrderedDict()

        if metric_per_sample:
            metrics = (sorted(self._per_sample_metrics, key=lambda i: i['sample_id']), metrics)

        accumulated_stats = \
            process_accumulated_stats(accumulated_stats=self._accumulated_layer_stats,
                                      stat_names_aliases=stat_names_aliases)

        self._accumulated_layer_stats = {}
        self._per_sample_metrics.clear()
        self.dump_prediction_to_annotation = False

        if stats_layout:
            restore_original_node_names(output_to_node_names, accumulated_stats, stats_layout, stat_aliases)

        return metrics, accumulated_stats

    def _load_model(self, paths):
        """ Loads IR model from disk
        :param paths: list of dictionaries:
        'name': name of the model (only for cascaded models)
        'model': path to the .xml model file,
        'weights': path to the .bin weights file
        :return list of dictionaries:
        'name': name of the model (only for cascaded models)
        'model': IE model instance
        """
        self._model_evaluator.load_network_from_ir(paths)
        return self._model_evaluator.get_network()

    def _ac_callback(self, stats_layout, metric_per_sample, value, **kwargs):
        if not ('metrics_result' in kwargs and 'dataset_indices' in kwargs):
            raise Exception('Expected "metrics_result", "dataset_indices" be passed to '
                            'output_callback inside accuracy checker')

        per_sample_metrics = kwargs['metrics_result']
        if metric_per_sample and per_sample_metrics is not None:
            for sample_id, metric_results in per_sample_metrics.items():
                for metric_result in metric_results:
                    self._per_sample_metrics.append(
                        {'sample_id': sample_id,
                         'metric_name': metric_result.metric_name,
                         'result': metric_result.result})
        if not stats_layout:
            return
        dataset_index = kwargs['dataset_indices'][0]
        append_stats(self._accumulated_layer_stats, stats_layout, value, dataset_index)

    @staticmethod
    def _set_requests_number(params, requests_number):
        if requests_number:
            params['nreq'] = np.clip(requests_number, 1, multiprocessing.cpu_count())
            if params['nreq'] != requests_number:
                logger.warning('Number of requests {} is out of range [1, {}]. Will be used {}.'
                               .format(requests_number, multiprocessing.cpu_count(), params['nreq']))

    def get_metrics_attributes(self):
        """Returns a dictionary of metrics attributes {metric_name: {attributes}}"""
        dataset_tag = self._evaluation_dataset_tag
        self._model_evaluator.select_dataset(dataset_tag)

        return self._model_evaluator.get_metrics_attributes()

    def add_metric(self, metric_config):
        self._model_evaluator.register_metric(metric_config)

    def add_postprocessing(self, postprocessing_config):
        self._model_evaluator.register_postprocessor(postprocessing_config)

    @property
    def evaluation_dataset_tag(self):
        return self._evaluation_dataset_tag

    @property
    def optimization_dataset_tag(self):
        return self._optimization_dataset_tag
