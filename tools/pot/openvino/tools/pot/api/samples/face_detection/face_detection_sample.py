# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser
from functools import partial
from time import time

import copy
import cv2
import numpy as np

from openvino.runtime import PartialShape    # pylint: disable=E0611,E0401
from openvino.tools.pot import Metric, DataLoader, IEEngine, \
    load_model, compress_model_weights, create_pipeline
from openvino.tools.pot.graph.model_utils import add_outputs
from openvino.tools.pot.samplers.batch_sampler import BatchSampler
from openvino.tools.pot.engines.utils import process_accumulated_stats, \
    restore_original_node_names, align_stat_names_with_results, \
    add_tensor_names, collect_model_outputs
from openvino.tools.pot.utils.logger import init_logger, get_logger
from openvino.tools.pot.api.samples.face_detection import utils

# Initialize the logger to print the quantization process in the console.
init_logger(level='INFO')
logger = get_logger(__name__)


# Custom DataLoader class implementation that is required for
# the proper reading of WIDER FACE images and annotations.
class WiderFaceLoader(DataLoader):

    # Required methods:
    def __init__(self, config):
        super().__init__(config)
        self._min_height_ann = 60
        self._img_ids, self._annotations = self._read_image_ids_annotations(self.config.annotation_file)

    def __getitem__(self, index):
        """
        Returns annotation and image (and optionally image metadata) at the specified index.
        Possible formats:
        (img_id, img_annotation), image
        (img_id, img_annotation), image, image_metadata
        """
        if index >= len(self):
            raise IndexError

        return self._read_image(self._img_ids[index]), self._annotations[self._img_ids[index]]

    def __len__(self):
        """ Returns size of the dataset """
        return len(self._img_ids)

    # Methods specific to the current implementation
    def _read_image_ids_annotations(self, annotation_file):
        with open(annotation_file) as f:
            content = f.read().split('\n')
        image_ids = [image_id for image_id, line in enumerate(content) if '.jpg' in line]
        annotations = {}
        image_names = []
        for image_id in image_ids:
            img_name = content[image_id]
            image_names.append(img_name)
            annotations[img_name] = []
            bbox_count = int(content[image_id + 1])
            bbox_list = content[image_id + 2:image_id + 2 + bbox_count]
            difficult_bboxes = []
            annotation_bboxes = []
            for idx, bbox in enumerate(bbox_list):
                bbox = bbox.strip(' ')
                x_min, y_min, width, height, *_ = np.array(bbox.split(' ')).astype(int)
                x_max, y_max = x_min + width, y_min + height
                if height < self._min_height_ann:
                    difficult_bboxes.append(idx)
                annotation_bboxes.append([x_min, y_min, x_max, y_max])
            annotations[img_name] = {'bboxes': annotation_bboxes,
                                     'difficult': difficult_bboxes}

        return image_names, annotations

    def _read_image(self, img_id):
        """ Reads image from directory.
        :param img_id: image id to read
        :return ndarray representation of image
        """
        image = cv2.imread(os.path.join(self.config.data_source, img_id))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class MTCNNEngine(IEEngine):
    def __init__(self, config, data_loader=None, metric=None):
        super().__init__(config, data_loader, metric)
        self._stages = [self._pnet_infer,
                        self._rnet_infer,
                        self._onet_infer]

    def set_model(self, model):
        """ Loads NetworkX model into InferenceEngine and stores it in Engine class
        :param model: CompressedModel instance
        """
        # save graph to IR and use it to initialize IE Network
        self._model = self._set_model(model)
        self._output_layers = {}
        stage_names = ['pnet', 'rnet', 'onet']
        for stage, _ in enumerate(model.models):
            self._output_layers[stage_names[stage]] = {
                'probabilities':  self.config['outputs']['probabilities'][stage],
                'regions':  self.config['outputs']['regions'][stage],
            }

    def _add_outputs(self, nodes_name):
        return add_outputs(self._model, nodes_name)

    def predict(self, stats_layout=None, sampler=None, stat_aliases=None,
                metric_per_sample=False, print_progress=False):
        stat_names_aliases = None
        if sampler is None:
            sampler = BatchSampler(self.data_loader)
        if stats_layout:
            model_with_stat_op, nodes_names_map, node_to_result_names = self._statistic_graph_builder. \
                insert_statistic(copy.deepcopy(self._nx_model),
                                 stats_layout, stat_aliases)
            self.set_model(model_with_stat_op)

            nodes_name = []
            for names_map in nodes_names_map.values():
                nodes_name.extend(list(names_map.keys()))

            outputs = self._add_outputs(nodes_names_map)
            for model_name, outputs_data in outputs.items():
                add_tensor_names(outputs_data, nodes_names_map[model_name].keys())

            model_output_names = []
            for model in self._model:
                model_output_names.extend(collect_model_outputs(model['model']))

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

        accumulated_stats = \
            process_accumulated_stats(stat_names_aliases=stat_names_aliases,
                                      accumulated_stats=self._accumulated_layer_stats)

        if stats_layout and stat_aliases:
            restore_original_node_names(node_to_result_names, accumulated_stats, stats_layout, stat_aliases)

        metrics = None
        if self._metric:
            metrics = self._metric.avg_value
            if metric_per_sample:
                metrics = (sorted(self._per_sample_metrics, key=lambda i: i['sample_id']), metrics)

        self._reset()

        return metrics, accumulated_stats

    def _predict(self, stats_layout, sampler, print_progress=False,
                 need_metrics_per_sample=False):
        progress_log_fn = logger.info if print_progress else logger.debug
        progress_log_fn('Start inference of %d images', len(sampler))

        # Start inference
        start_time = time()
        for batch_id, batch in iter(enumerate(sampler)):
            batch_annotations, image_batch, _ = self._process_batch(batch)

            for image in image_batch:
                stats_collect_callback = \
                    partial(self._collect_statistics,
                            stats_layout=stats_layout,
                            annotations=batch_annotations) if stats_layout else None

                result = None
                for stage in self._stages:
                    result = stage(image, result, stats_collect_callback)
                    if np.size(result) == 0:
                        break

                # Update metrics
                if np.size(result) != 0:
                    self._update_metrics(output=[result], annotations=batch_annotations,
                                         need_metrics_per_sample=need_metrics_per_sample)

            # Print progress
            if self._print_inference_progress(progress_log_fn,
                                              batch_id, len(sampler),
                                              start_time, time()):
                start_time = time()
        progress_log_fn('Inference finished')

    def _infer(self, data, ie_network, stats_collect_callback=None):
        ie_network.reshape(PartialShape(data.shape))
        filled_input = self._fill_input(ie_network, data)
        compiled_model = self._ie.compile_model(model=ie_network,
                                                device_name=self.config.device)
        infer_request = compiled_model.create_infer_request()
        result = infer_request.infer(filled_input)
        # Collect statistics
        if stats_collect_callback:
            stats_collect_callback(self._transform_for_callback(result))

        return result

    @staticmethod
    def _transform_for_callback(result):
        batch_size = len(list(result.values())[0])
        if batch_size == 1:
            return result
        return [{key: np.expand_dims(value[i], axis=0) for key, value in result.items()}
                for i in range(batch_size)]

    def _pnet_infer(self, image, _, stats_collect_callback):
        def preprocess(img):
            # Build an image pyramid
            img_pyramid, scales = utils.build_image_pyramid(img, 0.79, 1.2)
            return img_pyramid, {'scales': scales}

        def postprocess(output):
            # extract_predictions
            total_boxes = np.zeros((0, 9), float)
            for idx, outputs in enumerate(output):
                scales = input_meta['scales'][idx]
                mapping = outputs[[i for i, _ in outputs.items()
                                   if i.any_name == self._output_layers['pnet']['probabilities']][0]][0, 1]

                regions = outputs[[i for i, _ in outputs.items()
                                   if i.any_name == self._output_layers['pnet']['regions']][0]][0]

                boxes = utils.generate_bounding_box(mapping, regions, scales, 0.6)
                if len(boxes) != 0:
                    pick = utils.nms(boxes, 0.5)
                    if np.size(pick) > 0:
                        boxes = boxes[pick]
                if len(boxes) != 0:
                    total_boxes = np.concatenate((total_boxes, boxes), axis=0)

            if np.size(total_boxes) == 0:
                return np.zeros((0, 5))

            pick = utils.nms(total_boxes, 0.7)
            total_boxes = total_boxes[pick]

            return utils.bbreg(total_boxes[:, :5], total_boxes[:, 5:], include_bound=False)

        ie_network = self._model[0]['model']
        results = []
        image_pyramid, input_meta = preprocess(image)
        for data in image_pyramid:
            results.append(self._infer(data, ie_network, stats_collect_callback))
        # Process model output
        return postprocess(results)

    def _rnet_infer(self, image, prev_stage_output, stats_collect_callback):
        def preprocess(img):
            input_size = 24
            img = utils.cut_roi(img, prev_stage_output, input_size)
            return np.transpose(img, [0, 3, 2, 1])

        def postprocess(output):
            score = output[[i for i, _ in output.items()
                            if i.any_name == self._output_layers['rnet']['probabilities']][0]][:, 1]
            regions = output[[i for i, _ in output.items() if i.any_name == self._output_layers['rnet']['regions']][0]]
            return utils.calibrate_bboxes(prev_stage_output, score, regions, nms_type='union')

        ie_network = self._model[1]['model']
        data = preprocess(image)
        result = self._infer(data, ie_network, stats_collect_callback)
        return postprocess(result)

    def _onet_infer(self, image, prev_stage_output, stats_collect_callback):
        def preprocess(img):
            input_size = 48
            img = utils.cut_roi(img, prev_stage_output, input_size, False)
            return np.transpose(img, [0, 3, 2, 1])

        def postprocess(output):
            score = output[[i for i, _ in output.items()
                            if i.any_name == self._output_layers['onet']['probabilities']][0]][:, 1]
            regions = output[[i for i, _ in output.items() if i.any_name == self._output_layers['onet']['regions']][0]]
            bboxes = utils.calibrate_bboxes(prev_stage_output, score, regions)
            pick = utils.nms(bboxes, 0.7, 'min')
            bboxes_to_remove = np.setdiff1d(np.arange(len(bboxes)), pick)
            return np.delete(bboxes, bboxes_to_remove, axis=0)

        ie_network = self._model[2]['model']
        data = preprocess(image)
        result = self._infer(data, ie_network, stats_collect_callback)
        return postprocess(result)


# Custom implementation of Recall metric.
class Recall(Metric):

    # Required methods
    def __init__(self):
        super().__init__()
        self._name = 'recall'
        self._true_positives = []
        self._n_recorded_faces = []
        self._n_total_preds = []

    @property
    def avg_value(self):
        """ Returns average metric value for all model outputs.
        Possible format: {metric_name: metric_value}
        """
        n_total_preds = np.sum(self._n_total_preds)
        n_recorded_faces = np.sum(self._n_recorded_faces)
        tp = np.cumsum(np.concatenate(self._true_positives))[np.arange(n_total_preds)]
        recalls = tp / np.maximum(n_recorded_faces, np.finfo(np.float64).eps)
        return {self._name: recalls[-1]}

    def update(self, output, target):
        """ Calculates and updates metric value
        :param output: model output
        :param target: annotations
        """
        tps = []
        n_faces, n_preds = 0, 0
        for prediction, annotation in zip(output, target):
            n_preds += len(prediction)
            tp, n = self._calculate_tp(prediction, annotation)
            tps.extend(tp)
            n_faces += n
        self._true_positives.append(tps)
        self._n_recorded_faces.append(n_faces)
        self._n_total_preds.append(n_preds)

    def reset(self):
        """ Resets metric """
        self._true_positives = []
        self._n_recorded_faces = []
        self._n_total_preds = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'recall'}}

    # Methods specific to the current implementation
    @staticmethod
    def _calculate_tp(prediction, annotation):
        return utils.calculate_tp(prediction, annotation, overlap_threshold=0.5)


def main():
    parser = ArgumentParser(description='Post-training Compression Toolkit '
                                        'Face Detection Sample')
    parser.add_argument('-pm', '--pnet-model', help='Path to .xml of proposal network', required=True)
    parser.add_argument('-pw', '--pnet-weights', help='Path to .bin of proposal network')
    parser.add_argument('-rm', '--rnet-model', help='Path to .xml of refine network', required=True)
    parser.add_argument('-rw', '--rnet-weights', help='Path to .bin of refine network')
    parser.add_argument('-om', '--onet-model', help='Path to .xml of output network', required=True)
    parser.add_argument('-ow', '--onet-weights', help='Path to .bin of output network')
    parser.add_argument('-d', '--dataset', help='Path to the directory with images', required=True)
    parser.add_argument('-a', '--annotation-file',
                        help='File with WIDER FACE annotations in .txt format', required=True)

    args = parser.parse_args()

    model_config = {
        'model_name': 'mtcnn',
        'cascade': [
            {
                'name': 'pnet',
                'model': os.path.expanduser(args.pnet_model),
                'weights': os.path.expanduser(args.pnet_weights if args.pnet_weights else
                                              args.pnet_model.replace('.xml', '.bin'))
            },
            {
                'name': 'rnet',
                'model': os.path.expanduser(args.rnet_model),
                'weights': os.path.expanduser(args.rnet_weights if args.rnet_weights else
                                              args.rnet_model.replace('.xml', '.bin'))
            },
            {
                'name': 'onet',
                'model': os.path.expanduser(args.onet_model),
                'weights': os.path.expanduser(args.onet_weights if args.onet_weights else
                                              args.onet_model.replace('.xml', '.bin'))
            }
        ]
    }

    engine_config = {
        'device': 'CPU',
        'outputs': {
            'probabilities': ['prob1', 'prob1', 'prob1'],
            'regions': ['conv4-2', 'conv5-2', 'conv6-2']
        }
    }

    dataset_config = {
        'data_source': os.path.expanduser(args.dataset),
        'annotation_file': os.path.expanduser(args.annotation_file)
    }

    algorithms = [
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': 'ANY',
                'preset': 'performance',
                'stat_subset_size': 300
            }
        }
    ]

    # Step 1: Load the model.
    model = load_model(model_config)

    # Step 2: Initialize the data loader.
    data_loader = WiderFaceLoader(dataset_config)

    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
    metric = Recall()

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = MTCNNEngine(config=engine_config,
                         data_loader=data_loader,
                         metric=metric)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms, engine)

    # Step 6: Execute the pipeline.
    compressed_model = pipeline.run(model)

    # Step 7 (Optional): Compress model weights to quantized precision
    #                    in order to reduce the size of final .bin file.
    compress_model_weights(compressed_model)

    # Step 8: Save the compressed model to the desired path.
    compressed_model.save(os.path.join(os.path.curdir, 'optimized'))

    # Step 9 (Optional): Evaluate the compressed model. Print the results.
    metric_results = pipeline.evaluate(compressed_model)
    if metric_results:
        for name, value in metric_results.items():
            print('{: <27s}: {}'.format(name, value))


if __name__ == '__main__':
    main()
