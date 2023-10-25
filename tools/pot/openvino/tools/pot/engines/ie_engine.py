# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
from math import ceil
from time import time

import copy
import numpy as np
from openvino.runtime import Core, AsyncInferQueue, Shape   # pylint: disable=E0611,E0401

from .utils import append_stats, process_accumulated_stats, \
    restore_original_node_names, align_stat_names_with_results, \
    add_tensor_names, cast_friendly_names, collect_model_outputs, \
    process_raw_output, get_clean_name
from ..api.engine import Engine
from ..graph.model_utils import save_model
from ..samplers.batch_sampler import BatchSampler
from ..utils.logger import get_logger
from ..utils.utils import create_tmp_dir, convert_output_key

logger = get_logger(__name__)


# pylint: disable=W0631
class IEEngine(Engine):

    def __init__(self, config, data_loader=None, metric=None):
        super().__init__(config, data_loader, metric)
        self._ie = Core()
        self._ie.set_property({"ENABLE_MMAP": "NO"})
        self._model = None
        self._nx_model = None
        self._output_layers = None
        self._accumulated_layer_stats = dict()
        self._per_sample_metrics = []
        self._tmp_dir = create_tmp_dir()
        self._device = self.config.device

    def set_model(self, model):
        """ Loads NetworkX model into InferenceEngine and stores it in Engine class
        :param model: CompressedModel instance
        """
        if model.is_cascade:
            raise Exception('Cascade models are not supported in current engine')

        # save NetworkX graph to IR and use it to initialize IE Network
        self._model = self._set_model(model)[0]['model']
        self._output_layers = [get_clean_name(output.get_node().friendly_name) for output in self._model.outputs]

    def _set_model(self, model):
        """Creates IENetwork instances from NetworkX models in CompressedModel.
        :param: model: CompressedModel instance
        :return: list of dictionaries:
                 [
                    {
                        'name': model name (if model.is_cascaded),
                        'model': IENetwork instance
                    },
                ]
        """
        self._nx_model = model
        paths = save_model(model, self._tmp_dir.name, 'tmp_model', for_stat_collection=True)
        ie_networks = []
        for path_dict in paths:
            ie_net = {'model': self._ie.read_model(model=path_dict['model'], weights=path_dict['weights'])}
            if 'name' in path_dict:
                ie_net.update(name=path_dict['name'])
            ie_networks.append(ie_net)
        return ie_networks

    def predict(self, stats_layout=None, sampler=None, stat_aliases=None,
                metric_per_sample=False, print_progress=False):
        """ Performs model inference on specified dataset subset
         :param stats_layout: dict of stats collection functions {node_name: {stat_name: fn}} (optional)
         :param sampler: sampling dataset to make inference
         :param stat_aliases: dict of algorithms collections stats
                {algorithm_name: {node_name}: {stat_name}: fn} (optional)
         :param metric_per_sample: if Metric is specified and the value is True,
                then the metric value will be calculated for each data sample, otherwise for the whole dataset
         :param print_progress: whether to print inference progress
         :returns a tuple of dictionaries of persample and overall metric values if 'metric_per_sample' is True
                  ({sample_id: sample index, 'metric_name': metric name, 'result': metric value},
                   {metric_name: metric value}), otherwise, a dictionary of overall metrics
                   {metric_name: metric value}
                  a dictionary of collected statistics {node_name: {stat_name: [statistics]}}
        """

        if self._model is None:
            raise Exception('Model was not set in Engine class')

        # If sampler is not specified, make a prediction on the whole dataset
        if sampler is None:
            sampler = BatchSampler(self.data_loader)

        stat_names_aliases = None
        if stats_layout:
            model_with_stat_op, nodes_names_map, node_to_result_names = self._statistic_graph_builder.\
                insert_statistic(copy.deepcopy(self._nx_model),
                                 stats_layout, stat_aliases)
            self.set_model(model_with_stat_op)
            nodes_names_map = nodes_names_map[self._model.friendly_name]
            nodes_name = list(nodes_names_map.keys())
            cast_friendly_names(self._model.outputs)

            outputs = self._add_outputs(list(nodes_names_map.values()))
            add_tensor_names(outputs, nodes_name)

            model_output_names = collect_model_outputs(self._model)

            align_stat_names_with_results(model_output_names,
                                          nodes_name,
                                          node_to_result_names,
                                          stats_layout,
                                          stat_aliases)

            # Creating statistics layout with IE-like names
            stats_layout, stat_names_aliases = self._convert_stats_names(stats_layout)

        self._predict(stats_layout=stats_layout,
                      sampler=sampler,
                      print_progress=print_progress,
                      need_metrics_per_sample=metric_per_sample)

        # Process accumulated statistics
        # Replace IE-like statistics names with the original ones
        accumulated_stats = \
            process_accumulated_stats(accumulated_stats=self._accumulated_layer_stats,
                                      stat_names_aliases=stat_names_aliases)

        if stats_layout and stat_aliases:
            restore_original_node_names(node_to_result_names, accumulated_stats, stats_layout, stat_aliases)

        # Calculate metrics of required type. Reset collected statistics
        metrics = None
        if self._metric:
            metrics = self._metric.avg_value
            if metric_per_sample:
                metrics = (sorted(self._per_sample_metrics, key=lambda i: i['sample_id']), metrics)

        self._reset()

        return metrics, accumulated_stats

    @staticmethod
    def postprocess_output(outputs, _metadata):
        """ Processes model output data using the image metadata obtained during data loading
        :param outputs: dictionary of output data per output name
        :param _metadata: metadata obtained during data loading
        :return: list of the output data in an order expected by the accuracy metric if any is used
        """
        return list(outputs.values())

    def _reset(self):
        """ Resets collected statistics """
        if self._metric:
            self._metric.reset()
        self._per_sample_metrics = []
        self._accumulated_layer_stats = {}

    def _add_outputs(self, nodes_name):
        return self._model.add_outputs(nodes_name)

    def _predict(self, stats_layout, sampler, print_progress=False,
                 need_metrics_per_sample=False):
        """Performs model inference synchronously or asynchronously"""
        requests_number = self._get_requests_number(stats_layout)

        if requests_number == 1:
            self._process_dataset(stats_layout=stats_layout,
                                  sampler=sampler,
                                  print_progress=print_progress,
                                  need_metrics_per_sample=need_metrics_per_sample)
        else:
            self._process_dataset_async(stats_layout=stats_layout,
                                        sampler=sampler,
                                        print_progress=print_progress,
                                        need_metrics_per_sample=need_metrics_per_sample,
                                        requests_num=requests_number)

    def _process_infer_output(self, stats_layout, predictions,
                              batch_annotations, batch_meta, need_metrics_per_sample):
        # Collect statistics
        if stats_layout:
            self._collect_statistics(outputs=predictions,
                                     stats_layout=stats_layout,
                                     annotations=batch_annotations)

        # Postprocess network output
        processed_outputs = process_raw_output(predictions)
        outputs = {name: processed_outputs[name] for name in self._output_layers}
        logits = self.postprocess_output(outputs, batch_meta)

        # Update metrics
        if batch_annotations:
            self._update_metrics(output=logits, annotations=batch_annotations,
                                 need_metrics_per_sample=need_metrics_per_sample)

    def _collect_statistics(self, outputs, stats_layout, annotations=None):
        """Collects statistics of specified layers.
        :param outputs: layer outputs
        :param stats_layout: dict of stats collection functions {layer_name: [fn]}
        :param annotations: list of annotations [(img_id, annotation)]
        """
        dataset_index = annotations[0][0] if annotations is not None and annotations[0][0] else 0
        append_stats(self._accumulated_layer_stats, stats_layout, outputs, dataset_index)

    def _update_metrics(self, output, annotations, need_metrics_per_sample=False):
        """ Updates metrics.
        :param output: network output
        :param annotations: a list of annotations for metrics collection [(img_id, annotation)]
        :param need_metrics_per_sample: whether to collect metrics for each batch
        """
        _, batch_annotations = map(list, zip(*annotations))
        annotations_are_valid = all(a is not None for a in batch_annotations)

        if self._metric and annotations_are_valid:
            self._metric.update(output, batch_annotations)
            if need_metrics_per_sample:
                batch_metrics = self._metric.value
                for metric_name, metric_value in batch_metrics.items():
                    for i, annotation in enumerate(annotations):
                        self._per_sample_metrics.append({'sample_id': annotation[0],
                                                         'metric_name': metric_name,
                                                         'result': metric_value[i]})

    def _fill_input(self, model, image_batch):
        """Matches network input name with corresponding input batch
        :param model: IENetwork instance
        :param image_batch: list of ndarray images or list with a dictionary of inputs mapping
        """
        input_info = model.inputs
        batch_dim = self.config.get('batch_dim', 0)

        def is_dynamic_input(input_blob):
            return input_blob.partial_shape.is_dynamic

        def input_dim(input_blob):
            return len(input_blob.partial_shape)

        def process_input(input_blob, input_data):
            is_sampler_batchfied = len(input_data) != 1
            is_loader_batchfied = input_dim(input_blob) == input_data[0].ndim

            if is_loader_batchfied:
                if input_data[0].shape[batch_dim] == 1:
                    input_data = [np.squeeze(d, batch_dim) for d in input_data]
                    is_loader_batchfied = False
            if not is_sampler_batchfied and not is_loader_batchfied:
                is_sampler_batchfied = True

            assert not (is_sampler_batchfied and is_loader_batchfied), (
                "Data have to be batchfied by either 'stat_batch_size' parameter "
                "in quantization algorithm "
                "or a '__getitem__' method of 'DataLoader' not both."
            )

            input_data_batched = np.concatenate(
                [np.expand_dims(i, batch_dim) for i in input_data], axis=batch_dim
            )
            input_data_batched = input_data_batched.squeeze()
            if is_sampler_batchfied:
                if len(input_data_batched.shape) > batch_dim and input_data_batched.shape[batch_dim] != len(input_data):
                    input_data_batched = np.expand_dims(input_data_batched, batch_dim)

            if is_dynamic_input(input_blob):
                return input_data_batched
            else:
                return np.reshape(input_data_batched, input_blob.shape)

        if isinstance(image_batch[0], dict):
            feed_dict = {}
            input_blobs = {get_clean_name(in_node.get_node().friendly_name): in_node for in_node in input_info}
            for input_name in image_batch[0].keys():
                input_blob = input_blobs[input_name]
                input_blob_name = self._get_input_any_name(input_blob)
                feed_dict[input_blob_name] = process_input(
                    input_blob, [data[input_name] for data in image_batch]
                )
                if input_dim(input_blob) != feed_dict[input_blob_name].ndim:
                    raise ValueError(
                        "Incompatible input dimension. "
                        f"Cannot infer dimension {feed_dict[input_blob_name].ndim} "
                        f"{Shape(feed_dict[input_blob_name].shape)} "
                        f"into {input_dim(input_blob)}. "
                        "Please make sure batch of input is properly configured."
                    )
            return feed_dict

        if len(input_info) == 1:
            input_blob = next(iter(input_info))
            input_blob_name = self._get_input_any_name(input_blob)
            image_batch = {input_blob_name: process_input(input_blob, image_batch)}
            if input_dim(input_blob) != image_batch[input_blob_name].ndim:
                raise ValueError(
                    "Incompatible input dimension. "
                    f"Cannot infer dimension {image_batch[input_blob_name].ndim} "
                    f"{Shape(image_batch[input_blob_name].shape)} "
                    f"into {input_dim(input_blob)}. "
                    "Please make sure batch of input is properly configured."
                )
            if not is_dynamic_input(input_blob) and Shape(image_batch[input_blob_name].shape) != input_info[0].shape:
                raise ValueError(f"Incompatible input shapes. "
                                 f"Cannot infer {Shape(image_batch[input_blob_name].shape)} into {input_info[0].shape}."
                                 f"Try to specify the layout of the model.")
            return image_batch

        if len(input_info) == 2:
            image_info_nodes = list(filter(
                lambda x: len(x.shape) == 2, input_info))

            if len(image_info_nodes) != 1:
                raise Exception('Two inputs networks must contain exactly one ImageInfo node')

            image_info_node = image_info_nodes[0]
            image_info_name = self._get_input_any_name(image_info_node)
            image_tensor_node = next(iter(filter(
                lambda x: x.get_any_name() != image_info_name, input_info)))
            image_tensor_name = image_tensor_node.get_any_name()

            image_tensor = (image_tensor_name, process_input(image_tensor_node, image_batch))
            if not is_dynamic_input(image_tensor_node) and \
                    Shape(image_tensor[1].shape) != image_tensor_node.shape:
                raise ValueError(f"Incompatible input shapes. "
                                 f"Cannot infer {Shape(image_tensor[1].shape)} into {image_tensor_node.shape}."
                                 f"Try to specify the layout of the model.")

            ch, height, width = image_batch[0].shape
            image_info = (image_info_name,
                          np.stack(np.array([(height, width, ch)] * len(image_batch)), axis=0))

            return dict((k, v) for k, v in [image_tensor, image_info])

        raise Exception('Unsupported number of inputs')

    def _get_requests_number(self, stats_layout):
        """Returns number of requests for inference
        :param stats_layout: dict of stats collection functions {layer_name: [fn]} or None
        :return: number of requests
        """
        if stats_layout:
            requests_number = self._stat_requests_number
        else:
            requests_number = self._eval_requests_number

        if requests_number:
            requests_number_clipped = np.clip(requests_number, 1, multiprocessing.cpu_count())
            if requests_number_clipped != requests_number:
                logger.warning('Number of requests {} is out of range [1, {}]. Will be used {}.'
                               .format(requests_number, multiprocessing.cpu_count(), requests_number_clipped))
                requests_number = requests_number_clipped
        else:
            requests_number = 0

        return requests_number

    def _process_dataset_async(self, stats_layout, sampler, print_progress=False,
                               need_metrics_per_sample=False, requests_num=0):
        """Performs model inference on specified dataset subset asynchronously
        :param stats_layout: dict of stats collection functions {node_name: [fn]}(optional)
        :param sampler: sampling dataset to make inference
        :param print_progress: whether to print inference progress
        :param need_metrics_per_sample: whether to collect metrics for each batch
        :param requests_num: number of infer requests
        """

        def completion_callback(request, user_data):
            start_time, batch_id, batch_annotations, batch_meta = user_data
            predictions = request.results
            self._process_infer_output(stats_layout, predictions,
                                       batch_annotations, batch_meta,
                                       need_metrics_per_sample)

            # Print progress
            if self._print_inference_progress(progress_log_fn,
                                              batch_id, len(sampler),
                                              start_time, time()):
                start_time = time()

        progress_log_fn = logger.info if print_progress else logger.debug
        self._ie.set_property(self._device,
                              {'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO', 'CPU_BIND_THREAD': 'YES'})
        # Load model to the plugin
        compiled_model = self._ie.compile_model(model=self._model, device_name=self._device)
        optimal_requests_num = compiled_model.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
        requests_num = optimal_requests_num if requests_num == 0 else requests_num
        logger.debug('Async mode requests number: %d', requests_num)
        infer_queue = AsyncInferQueue(compiled_model, requests_num)

        progress_log_fn('Start inference of %d images', len(sampler))

        sampler_iter = iter(enumerate(sampler))
        # Start inference
        start_time = time()
        infer_queue.set_callback(completion_callback)
        for batch_id, data_batch in sampler_iter:
            batch_annotations, image_batch, batch_meta = self._process_batch(data_batch)
            user_data = (start_time, batch_id, batch_annotations, batch_meta)
            infer_queue.start_async(self._fill_input(compiled_model, image_batch), user_data)
        infer_queue.wait_all()
        progress_log_fn('Inference finished')

    def _process_dataset(self, stats_layout, sampler, print_progress=False,
                         need_metrics_per_sample=False):
        """
        Performs model inference on specified dataset subset synchronously
        :param stats_layout: dict of stats collection functions {node_name: {stat_name: fn}} (optional)
        :param sampler: sampling dataset to make inference
        :param print_progress: whether to print inference progress
        :param need_metrics_per_sample: whether to collect metrics for each batch
        """

        progress_log_fn = logger.info if print_progress else logger.debug

        # Load model to the plugin
        compiled_model = self._ie.compile_model(model=self._model, device_name=self._device)
        infer_request = compiled_model.create_infer_request()

        progress_log_fn('Start inference of %d images', len(sampler))

        # Start inference
        start_time = time()
        for batch_id, batch in iter(enumerate(sampler)):
            batch_annotations, image_batch, batch_meta = self._process_batch(batch)

            # Infer batch of images
            predictions = infer_request.infer(self._fill_input(compiled_model, image_batch))
            outputs = predictions

            self._process_infer_output(stats_layout, outputs,
                                       batch_annotations, batch_meta,
                                       need_metrics_per_sample)

            # Print progress
            if self._print_inference_progress(progress_log_fn,
                                              batch_id, len(sampler),
                                              start_time, time()):
                start_time = time()

        progress_log_fn('Inference finished')

    @staticmethod
    def _process_batch(batch):
        """ Processes batch data and returns lists of annotations, images and batch meta data
        :param batch: a list with batch data.
                      Possible formats: [((img_id, label), image)]
                                        [({img_id: label}, image)]
        :returns a list with annotations [(img_id, label)]
                 a list with input data  [image]
                 a list with batch meta data
        """
        if not all([isinstance(item, tuple) for item in batch]):
            raise RuntimeError('Inconsistent data in the batch. '
                               'Some items contain annotation, and some do not.')

        if not all([isinstance(item[0], tuple) for item in batch]):
            images, image_annotation = [data[0] for data in batch], [(idx, data[1]) for idx, data in enumerate(batch)]
        else:
            images, image_annotation = [data[1] for data in batch], [data[0] for data in batch]

        if all([len(item) == 2 for item in batch]):
            meta_data = [{}]*len(images)
        elif all([len(item) == 3 for item in batch]):
            meta_data = [data[2] for data in batch]
        else:
            raise RuntimeError('Inconsistent data in the batch. '
                               'Some items contain meta data, and some do not.')

        # if image annotations are represented as dictionaries, convert them to tuples
        if image_annotation is not None and all([isinstance(item, dict) for item in image_annotation]):
            image_annotation = [(img_id, label) for sample_annot in image_annotation
                                for img_id, label in sample_annot.items()]

        return image_annotation, images, meta_data

    @staticmethod
    def _print_inference_progress(log_fn, current_id, total_dataset_size,
                                  start_time, finish_time):
        """Prints inference progress. Returns True if timer needs update"""
        if (current_id + 1) % ceil(total_dataset_size / 10) == 0:
            log_fn('%d/%d batches are processed in %.2fs',
                   current_id + 1,
                   total_dataset_size,
                   finish_time - start_time)
            return True
        return False

    @staticmethod
    def _convert_stats_names(stats_layout):
        """Converts statistics names from MO format to IE format"""
        stat_names_aliases = {convert_output_key(key): key for key in stats_layout}
        stats_layout_ie_style = {convert_output_key(key): value
                                 for key, value in stats_layout.items()}
        return stats_layout_ie_style, stat_names_aliases

    @staticmethod
    def _get_input_any_name(input_node):
        """ Returns any_name for nGraph input const node
            If any_name not exists, sets this name as friendly_name
            :param input_node - nGraph ConstOutput object
            :returns - a string tensor name
        """
        try:
            input_name = input_node.get_any_name()
        except RuntimeError:
            name_set = set([input_node.node.friendly_name])
            input_name = input_node.get_tensor().set_names(name_set)
            input_name = input_node.get_any_name()
        return input_name
