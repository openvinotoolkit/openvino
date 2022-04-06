# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np

from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....samplers.creator import create_sampler
from ....statistics.statistics import TensorStatistic
from ....statistics.functions import activations as asf
from ....statistics.functions import aggregation as agf
from ....utils.logger import get_logger


logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('ActivationChannelAlignment')
class ActivationChannelAlignment(Algorithm):
    name = 'ActivationChannelAlignment'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        stat_subset_size = min(
            self._config.get(
                'stat_subset_size', len(self._engine.data_loader)),
            len(self._engine.data_loader))
        self.total_exec_steps = stat_subset_size
        shuffle_data = self._config.get('shuffle_data', False)
        seed = self._config.get('seed', 0)
        self._sampler = create_sampler(engine, stat_subset_size, shuffle_data, seed)

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        """ this function applies activation range alignment procedure
         :param model: model to apply the algo on
         :return range-corrected model
         """

        activations_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)
        node_pairs_list = self.find_node_pairs(model)

        stats = dict()
        for node_name, stats_list in activations_statistics.items():
            stats[node_name] = dict()
            for stats_name, stats_values in stats_list.items():
                stats[node_name][stats_name] = agf.median(stats_values)

        for node_pair_data in node_pairs_list:
            node_in, weight_in, bias_in, node_out, weight_out, bias_out = node_pair_data
            self.align_ranges(node_in, weight_in, bias_in, weight_out, bias_out, stats)
            node_out['need_rescale'] = True

        return model

    def register_statistics(self, model, stats_collector):
        model = deepcopy(model)
        activation_statistics_layout = self.get_activations_statistics_layout(model)
        stats_collector.register(self.name, activation_statistics_layout, self._sampler)
        self._stats_collector = stats_collector

    def get_activations_statistics_layout(self, model):
        node_pairs_list = self.find_node_pairs(model)

        stats_layout = {}
        for node_pair_data in node_pairs_list:
            node_in, *_ = node_pair_data
            # Step over bias Add node
            if nu.get_bias_for_node(node_in):
                node_in = nu.get_node_output(node_in, 0)[0]
            name = node_in.fullname
            stats_layout[name] = {'channel_range_min': TensorStatistic(asf.quantile_per_channel, q=1e-4),
                                  'channel_range_max': TensorStatistic(asf.quantile_per_channel, q=1-1e-4)}

        logger.debug('Collecting output statistics for nodes {}'.format(stats_layout.keys()))
        return stats_layout

    def align_ranges(self, node_in, weight_in, bias_in, weight_out, bias_out, stats):
        # Step over bias Add node
        if nu.get_bias_for_node(node_in):
            node_in = nu.get_node_output(node_in, 0)[0]
        name = node_in.fullname

        amin = stats[name]['channel_range_min']
        amax = stats[name]['channel_range_max']
        ascale, amean = amax - amin, (amin + amax) * 0.5
        self.align_means(amean, bias_in, bias_out, weight_out)
        self.align_scales(ascale, bias_in, weight_in, weight_out)

    @staticmethod
    def align_scales(ascale, bias_in, weight_in, weight_out):
        if np.all(ascale <= 0):
            return

        scale_factor = ascale / np.median(ascale[ascale > 0])
        scale_factor[ascale <= 0] = 1
        scale_factor = np.clip(scale_factor, 1e-2, 1e2)

        # scale producer convolution weights
        weight_in_value = nu.get_node_value(weight_in)
        weight_in_type = weight_in_value.dtype
        weight_in_shape = weight_in_value.shape
        if weight_in_shape[0] == scale_factor.shape[0]:
            scale_in_shape = np.ones(len(weight_in_shape), dtype=np.int)
            scale_in_shape[0] = scale_factor.shape[0]
            weight_in_value = weight_in_value / scale_factor.reshape(scale_in_shape)
            nu.set_node_value(weight_in, weight_in_value.astype(weight_in_type))

            if bias_in is not None:
                # set new bias
                old_bias_in_val = nu.get_node_value(bias_in)
                old_bias_in_type = old_bias_in_val.dtype
                bias_in_value = old_bias_in_val / scale_factor.reshape(old_bias_in_val.shape)
                nu.set_node_value(bias_in, bias_in_value.astype(old_bias_in_type))

            weight_out_value = nu.get_node_value(weight_out)
            weight_out_type = weight_out_value.dtype
            scale_out_shape = np.ones(len(weight_out_value.shape), dtype=np.int)
            scale_out_shape[1] = scale_factor.shape[0]
            weight_out_value = weight_out_value * scale_factor.reshape(scale_out_shape)
            nu.set_node_value(weight_out, weight_out_value.astype(weight_out_type))

    @staticmethod
    def align_means(amean, bias_in, bias_out, weight_out):
        # shift biases to get 0 mean activations in each channel
        if bias_in and bias_out:
            old_bias_in_value = nu.get_node_value(bias_in)
            old_bias_in_type = old_bias_in_value.dtype
            bias_in_value = old_bias_in_value - amean.reshape(old_bias_in_value.shape)
            nu.set_node_value(bias_in, bias_in_value.astype(old_bias_in_type))

            weight_out_value = nu.get_node_value(weight_out)
            weight_dims = len(weight_out_value.shape)
            if weight_dims > 2:
                weight_out_value = np.sum(weight_out_value,
                                          axis=tuple(range(2, weight_dims)))
            shift = weight_out_value.dot(amean)
            del weight_out_value

            old_bias_out_value = nu.get_node_value(bias_out)
            old_bias_out_type = old_bias_out_value.dtype
            bias_out_value = old_bias_out_value + shift.reshape(old_bias_out_value.shape)
            nu.set_node_value(bias_out, bias_out_value.astype(old_bias_out_type))

    def find_node_pairs(self, model):
        node_pairs_list = []
        nodes = sorted([(n.fullname, n) for n in mu.get_nodes_by_type(model, ['Convolution'])])
        for _, node_out in nodes:
            if not self.check_conv_node(node_out):
                continue
            node_in = nu.get_node_input(node_out, 0)

            # Conv -> Add -> Conv
            if node_in.type == 'Add':
                node_in = nu.get_node_input(node_in, 0)

            if not self.check_producer_node(node_in, node_out):
                continue

            bias_in, weights_in = self.get_producer_weights(node_in)
            bias_out, weights_out = self.get_consumer_weights(node_out)

            if not weights_in or not weights_out:
                continue
            # node_in -> node_out
            node_pairs_list.append((node_in, weights_in, bias_in,
                                    node_out, weights_out, bias_out))
            # align activations channels inside this sequence
            logger.debug('{} -> {}'.format(node_in.fullname, node_out.fullname))

        return node_pairs_list

    @staticmethod
    def check_conv_node(node_out):
        if node_out.has_valid('group') and node_out.group != 1:
            return False
        if not node_out.has_valid('pads_begin') or not node_out.has_valid('pads_end') or \
                not np.all(np.array(node_out.pads_begin) == 0) or not np.all(np.array(node_out.pads_end) == 0):
            logger.debug('Pad of {} Convolution node != 0 '
                         'Do not align activations for this node pair.'.format(node_out.fullname))
            return False
        if not node_out.has_valid('strides') or not np.all(np.array(node_out.strides) == 1):
            logger.debug('Strides of {} Convolution node != 1 '
                         'Do not align activations for this node pair.'.format(node_out.fullname))
            return False
        if not node_out.has_valid('strides') or not np.all(np.array(node_out.dilations) == 1):
            logger.debug('Dilation of {} Convolution node != 1 '
                         'Do not align activations for this node pair.'.format(node_out.fullname))
            return False
        return True

    @staticmethod
    def check_producer_node(node_in, node_out):
        # look into producer. skip linear operations and check that
        # it is convolutions with single fq consumer
        node_out_producer_port = node_out.in_port(0).get_source()
        if len(node_out_producer_port.get_destinations()) > 1:
            logger.debug('{} has a producer that feeds many nodes. '
                         'Do not align activations for this node pair.'.format(node_out.fullname))
            return False
        # check that producer is convolution
        if node_in.type not in ['Convolution', 'MatMul']:
            logger.debug('{} gets data from {} {}'.format(node_out.fullname, node_in.fullname, node_in.type))
            logger.debug('{} has no Convolution producer. '
                         'Do not align activations for this node pair.'.format(node_out.fullname))
            return False
        return True

    @staticmethod
    def get_producer_weights(node_in):
        # get producer convolution weights
        w_in = nu.get_weights_for_node(node_in)
        if w_in.type == 'FakeQuantize':
            w_in = nu.get_node_input(w_in, 0)
        if w_in.type != 'Const':
            w_in = None
            logger.debug('Node after {} has no Convolution producer with const weights. '
                         'Do not align activations for this node pair.'.format(node_in))

        # get producer convolution bias
        b_in = nu.get_bias_for_node(node_in)
        if b_in is not None and b_in.type != 'Const':
            w_in = None
            logger.debug('Node after {} has no convolution producer with const bias. '
                         'Do not align activations for this node pair.'.format(node_in))
        return b_in, w_in

    @staticmethod
    def get_consumer_weights(node_out):
        # get consumer convolution weights
        w_out = nu.get_weights_for_node(node_out)
        if w_out.type == 'FakeQuantize':
            w_out = nu.get_node_input(w_out, 0)
        if w_out.type != 'Const':
            w_out = None
            logger.debug('{} has no const weights. '
                         'Do not align activations for this node pair.'.format(node_out.fullname))

        # get consumer convolution bias
        b_out = nu.get_bias_for_node(node_out)
        if b_out is not None and b_out.type != 'Const':
            b_out = None
        return b_out, w_out
