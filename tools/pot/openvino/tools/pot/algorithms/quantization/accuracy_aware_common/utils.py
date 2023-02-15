# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from functools import partial

import numpy as np
import scipy
from addict import Dict

from ....algorithms.quantization import utils as eu
from ....engines.ac_engine import ACEngine
from ....graph.model_utils import get_nodes_by_type
from ....graph.node_utils import get_all_node_outputs
from ....graph.utils import find_operation_matches
from ....samplers.creator import create_sampler

SPECIAL_METRICS = ['cmc', 'reid_map', 'pairwise_accuracy_subsets', 'pairwise_accuracy', 'normalized_embedding_accuracy',
                   'face_recognition_tafa_pair_metric', 'localization_recall',
                   'coco_orig_keypoints_precision', 'coco_orig_segm_precision', 'coco_orig_keypoints_precision',
                   'spearman_correlation_coef', 'pearson_correlation_coef']

METRICS_CONFIGS = {'sigmoid_recom_loss': {'metrics': 'log_loss',
                                          'postprocessing': 'sigmoid_normalize_recommendation'},
                   'coco_precision': {'metrics': 'coco_precision'},
                   'coco_segm_precision': {'metrics': 'coco_segm_precision'}}

METRIC2PROXY_METRIC = {
    'hit_ratio':
        {
            'persample': 'sigmoid_recom_loss',
            'ranking': 'sigmoid_recom_loss'
        },
    'ndcg':
        {
            'persample': 'sigmoid_recom_loss',
            'ranking': 'sigmoid_recom_loss'
        },
    'coco_orig_precision':
        {
            'persample': 'coco_precision'
        },
    'coco_orig_keypoints_precision':
        {
            'persample': 'coco_precision'
        },
    'coco_orig_segm_precision':
        {
            'persample': 'coco_segm_precision'
        }
}


def create_metric_config(engine, algo_config: Dict, force_logit_comparison=False,
                         logit_distance_type='cosine') -> Dict:
    def create_metric_params(metric_name):
        engine_metrics_attributes = engine.get_metrics_attributes()
        if metric_name not in engine_metrics_attributes:
            RuntimeError('Couldn\'t create metric parameters. '
                         'Metric {} not registered in the engine.'.format(metric_name))
        params = Dict()
        params.name = metric_name
        params.type = engine_metrics_attributes[metric_name]['type']

        params.is_special = (params.type in SPECIAL_METRICS) or force_logit_comparison

        if engine_metrics_attributes[metric_name]['direction'] == 'higher-better':
            params.comparator = (lambda a: a)
        elif engine_metrics_attributes[metric_name]['direction'] == 'higher-worse':
            params.comparator = (lambda a: -a)
        else:
            raise ValueError('Unexpected {} metric direction value.'.format(metric_name))

        params.sort_fn = partial(sort_by_logit_distance, distance=logit_distance_type) \
            if params.is_special else partial(sort_by_metric_difference, comp_fn=params.comparator)

        return params

    def metric_to_proxy_map(metrics):
        """Determines which metrics need proxy metrics and creates metrics to proxy metrics map.
        :param metrics: optimizable metrics names
        :returns a dictionary of metrics to proxy metrics mapping {metric_name: 'persample': proxy_name,
                                                                                'ranking': proxy_name}
                 a list of proxy metrics names to register
        """

        def update_proxy_list(proxy_metric_name):
            """Updates a list of proxy metrics names to register.
            :return a proxy metric name in accordance with the engine naming
            """
            proxy_config = METRICS_CONFIGS.get(proxy_metric_name, {})
            metric_config = proxy_config.get('metrics')
            postprocessing_config = proxy_config.get('postprocessing')
            if metric_config or postprocessing_config:
                to_register.add(proxy_metric_name)
            return metric_name_from_config(metric_config)

        match_names_config = Dict({metric_name: {} for metric_name in metrics})
        to_register = set()

        for metric_name, metric_type in metrics:
            if metric_type in METRIC2PROXY_METRIC:
                persample_metric_name = METRIC2PROXY_METRIC[metric_type].get('persample')
                persample_proxy_metric_name = update_proxy_list(persample_metric_name)
                if persample_proxy_metric_name:
                    match_names_config[metric_name].persample = persample_proxy_metric_name

                ranking_metric_name = METRIC2PROXY_METRIC[metric_type].get('ranking')
                ranking_proxy_metric_name = update_proxy_list(ranking_metric_name)
                if ranking_proxy_metric_name:
                    match_names_config[metric_name].ranking = ranking_proxy_metric_name

        return match_names_config, list(to_register)

    metrics_attributes = engine.get_metrics_attributes()

    # configure which metrics to optimize
    if algo_config.metrics:
        metrics_names = []
        for metric in algo_config.metrics:
            metric_type = metric.type if metric.type else metric.name
            metrics_names.append((metric.name, metric_type))
    else:
        metrics_names = [(metric_name, metric_attr.get('type', metric_name)) for metric_name, metric_attr
                         in metrics_attributes.items()]

    # register proxy metrics
    metrics_to_proxy_map, metrics_to_register = metric_to_proxy_map(metrics_names)
    register_metrics(engine, metrics_to_register)

    metrics_config = Dict()
    for metric, _ in metrics_names:
        persample_name = metrics_to_proxy_map[metric].get('persample', metric)
        ranking_name = metrics_to_proxy_map[metric].get('ranking', metric)
        metrics_config[metric].persample = create_metric_params(persample_name)
        metrics_config[metric].ranking = create_metric_params(ranking_name)
        metrics_config[metric].update(create_metric_params(metric))

    return metrics_config


def metric_name_from_config(metric_config):
    if isinstance(metric_config, str):
        return metric_config
    if isinstance(metric_config, dict):
        return metric_config.get('name', metric_config['type'])
    return None


def register_metrics(engine, metrics_names: list):
    """Registers metrics and postprocessing in the engine.
    :param engine: an engine in which metrics will be registered
    :param metrics_names: a list of metrics names
    """
    registered_metrics = engine.get_metrics_attributes()
    for metric in metrics_names:
        if metric not in METRICS_CONFIGS:
            raise ValueError('Cannot register metric. Unsupported name {}.'.format(metric))
        proxy_config = METRICS_CONFIGS.get(metric, {})
        if 'metrics' in proxy_config:
            metric_config = proxy_config['metrics']
            if metric_name_from_config(metric_config) not in registered_metrics:
                register_metric(engine, metric_config)
        if 'postprocessing' in proxy_config:
            postprocessing_config = proxy_config['postprocessing']
            register_postprocessing(engine, postprocessing_config)


def sort_by_logit_distance(u, v, reverse=False, distance='cosine'):
    if len(u) != len(v):
        raise RuntimeError('Cannot compare samples. '
                           'Lists of per-sample metric results should be the same length.')

    kd_distance = lambda u, v: scipy.stats.entropy(scipy.special.softmax(u),
                                                   scipy.special.softmax(v))
    mse_distance = lambda u, v: np.mean((u - v) ** 2)

    nmse_distance = lambda u, v: np.dot(u - v, u - v) / np.dot(u, u)

    distance_function = {
        'cosine': scipy.spatial.distance.cosine,
        'kd': kd_distance,
        'mse': mse_distance,
        'nmse': nmse_distance,
    }

    distance_between_samples = np.array([distance_function[distance](ui.flatten(), vi.flatten())
                                         for ui, vi in zip(u, v)])
    sorted_arr = np.argsort(distance_between_samples)
    if reverse:
        sorted_arr = np.flip(sorted_arr)
    return sorted_arr


def sort_by_metric_difference(u, v, comp_fn=lambda a: a, reverse=False):
    if len(u) != len(v):
        raise RuntimeError('Cannot compare samples. '
                           'Lists of per-sample metric results should be the same length.')
    u = np.asarray(u)
    v = np.asarray(v)
    sorted_arr = np.argsort(comp_fn(u - v))
    if reverse:
        sorted_arr = np.flip(sorted_arr)
    return sorted_arr


def register_metric(engine, metric_config):
    if isinstance(engine, ACEngine):
        engine.add_metric(metric_config)
    else:
        raise NotImplementedError('{} engine cannot register new metrics.'
                                  .format(type(engine).__name__))


def register_postprocessing(engine, postprocessing_config):
    if isinstance(engine, ACEngine):
        engine.add_postprocessing(postprocessing_config)
    else:
        raise NotImplementedError('{} engine cannot register new postprocessing.'
                                  .format(type(engine).__name__))


def is_preset_performance(config: Dict):
    if config.weights.mode == 'symmetric' and config.activations.mode == 'symmetric':
        return True
    if config.weights.mode == 'asymmetric' or config.activations.mode == 'asymmetric':
        return False
    if config.preset == 'performance':
        return True
    return False


def get_mixed_preset_config(config: Dict):
    config = deepcopy(config)
    config.update(preset='mixed')
    if config.activations.mode:
        config.activations.mode = 'asymmetric'
    if config.weights.mode:
        config.weights.mode = 'symmetric'
    return config


def get_num_of_quantized_ops(model, quantizable_operations):
    quantized_ops = set()
    nodes_to_see = []
    for fq_node in get_nodes_by_type(model, ['FakeQuantize']):
        nodes_to_see.extend(get_all_node_outputs(fq_node))
        while nodes_to_see:
            child = nodes_to_see.pop()
            if find_operation_matches(quantizable_operations, child):
                quantized_ops.add(child)
                continue
            nodes_to_see.extend(get_all_node_outputs(child))

    return len(quantized_ops)


def evaluate_model(
        model, engine,
        dataset_size,
        subset_indices=None,
        print_progress=True,
        metrics_config=None,
        per_sample_subset_indices=None,
        output_node_name=None,
        stats_layout=None,
):
    """Evaluates the model and processes metrics values
        :param model: model to evaluate
        :param subset_indices: image indices to evaluate on. If None evaluate on whole dataset
        :param per_sample_subset_indices: image indices for which to return per-sample metrics.
                                          If None for all predicted images
        :param print_progress: Whether to print inference progress
        :returns a dictionary of predicted metrics {metric_name: value}
                 a dictionary of per-sample metrics values {metric_name: [values]}
    """
    engine.set_model(model)
    eu.select_evaluation_dataset(engine)

    if not subset_indices:
        subset_indices = range(dataset_size)

    index_sampler = create_sampler(engine, samples=subset_indices)
    (metrics_per_sample, metrics), raw_output = engine.predict(stats_layout=stats_layout,
                                                               sampler=index_sampler,
                                                               metric_per_sample=True,
                                                               print_progress=print_progress)

    raw_output = process_raw_output(raw_output, output_node_name)
    metrics_per_sample = process_per_sample_metrics(metrics_per_sample,
                                                    metrics_config,
                                                    per_sample_subset_indices,
                                                    raw_output=raw_output)
    metrics = dict((name, value) for name, value in metrics.items() if name in metrics_config)
    eu.reset_dataset_to_default(engine)

    return metrics, metrics_per_sample


def process_raw_output(output, output_node_name):
    if not output:
        return []
    return output[output_node_name]['output_logits']


def process_per_sample_metrics(metrics_per_sample, metrics_config,
                               indices=None, raw_output=None):
    """Creates a dictionary of per-sample metrics values {metric_name: [values]}
            :param metrics_per_sample: list of per-sample metrics
            :param indices: indices of samples to be considered. All if None
            :param raw_output: raw output from the model
            :return processed dictionary
            """
    metrics_to_keep = {config.persample.name: config.persample
                       for config in metrics_config.values()}

    if not metrics_to_keep:
        return {}

    processed_metrics_per_sample = dict((name, []) for name in metrics_to_keep)

    for metric_name, metric_params in metrics_to_keep.items():
        if metric_params.is_special:
            processed_metrics_per_sample[metric_name] = raw_output

    for value in metrics_per_sample:
        if value['metric_name'] in metrics_to_keep:
            if metrics_to_keep[value['metric_name']].is_special:
                continue
            if value['result'] is not None:
                result_value = np.nanmean(value['result'])
            else:
                result_value = None
            processed_metrics_per_sample[value['metric_name']].append(result_value)

    # check that all metrics have equal number of samples
    if not len({len(value) for value in processed_metrics_per_sample.values()}) == 1:
        raise RuntimeError('Inconsistent number of per-sample metric values')

    if indices:
        for name, values in processed_metrics_per_sample.items():
            processed_metrics_per_sample[name] = [values[i] for i in indices]

    return processed_metrics_per_sample


def prepare_nodes_for_logger(nodes_names):
    postprocessed_nodes_names = []
    for name in nodes_names:
        subgraphs = name.split('|')
        if len(subgraphs) > 1:
            postprocessed_nodes_names.append(str(subgraphs))
        else:
            postprocessed_nodes_names.append(subgraphs[0])
    return postprocessed_nodes_names
