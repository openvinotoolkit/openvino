import os
import random
from pathlib import PosixPath
from collections import OrderedDict
import pytest
import numpy as np
import cv2

from openvino.tools.pot import load_model
from openvino.tools.pot import Metric, DataLoader, IEEngine, \
    load_model, save_model, compress_model_weights, create_pipeline

from .utils.config import get_engine_config, merge_configs, \
    get_dataset_info, PATHS2DATASETS_CONFIG, make_algo_config

TEST_MODELS = [
    (
        'squeezenet1.1',
        'caffe',
        'AccuracyAwareQuantization',
        'performance',
        {'accuracy@top1': 0.410},
        {
            'num_samples': 100,
            'image_shape': (227, 227, 3),
            'seed': 0,
        },
        {
            'target_device': 'GNA',
            'preset': 'performance',
            'stat_subset_size': 100,
            'seed': 0,
            'maximal_drop': 0.01,
            'max_iter_num': 1
        }
    ),
    (
        'hbonet-0.25',
        'pytorch',
        'AccuracyAwareQuantization',
        'accuracy',
        {'accuracy@top1': 0.48},
        {
            'num_samples': 100,
            'image_shape': (224, 224, 3),
            'seed': 0,
        },
        {
            'target_device': 'GNA3',
            'preset': 'performance',
            'stat_subset_size': 10,
            'seed': 0,
            'maximal_drop': 0.5,
            'max_iter_num': 1
        }
    )
]


class SyntheticDataLoader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self._imgs = self._generate_synthetic_data(self.config)
        self._annotations = {}

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, index):
        if len(self._annotations) == 0:
            return self._imgs[index], index
        else:
            return self._imgs[index], self._annotations[index]

    def _generate_synthetic_data(self, config):
        imgs = []
        np.random.seed(config.seed)
        for i in range(config.num_samples):
            img = np.random.randint(low=0, high=255, size=config.image_shape, dtype=np.uint8)
            imgs.append(img)

        return imgs

    def _set_annotations(self, annotations):
        self._annotations = annotations

    def get_annotations(self):
        return self._annotations


class LabelCollector(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'labelcollector'
        self._annotations = {}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        return {self._name: 1.0}

    def update(self, outputs, targets):
        for output, target in zip(outputs, targets):
            if isinstance(target, dict):
                target = list(target.values())
            pred = np.argmax(output, axis=1).item()
            self._annotations[target] = pred

    def reset(self):
        pass

    def get_annotations(self):
        return self._annotations.copy()

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'accuracy'}}


class Accuracy(Metric):
    # Custom implementation of classification accuracy metric.
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = 'accuracy@top{}'.format(self._top_k)
        self._matches = []

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        """ Updates prediction matches.
        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception('The accuracy metric cannot be calculated '
                            'for a model with multiple outputs')
        if isinstance(target, dict):
            target = list(target.values())
        predictions = np.argsort(output[0], axis=1)[:, -self._top_k:]
        match = [float(t in predictions[i]) for i, t in enumerate(target)]

        self._matches.append(match)

    @property
    def value(self):
        """ Returns metric value for the last model output.
         Possible format: {metric_name: [metric_values_per_image]}
         """
        return {self._name: [np.ravel(self._matches[-1]).mean()]}

    def reset(self):
        """ Resets collected matches """
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'accuracy'}}


def get_engine_configs():
    engine_config = {
        'device': 'CPU',
        'stat_requests_number': 1,
        'eval_requests_number': 1,
    }
    return engine_config


@pytest.fixture(scope='module', params=TEST_MODELS,
                ids=['{}_{}_{}_{}'.format(*m) for m in TEST_MODELS])
def _params(request):
    return request.param


def get_model(models, model_name, model_framework, tmp_path):
    """
    Returns a model to test by downloading from omz
    """
    model = models.get(model_name, model_framework, tmp_path)
    model_params = model.model_params
    model_config = {
        'model_name': 'sample_model',
        'model': os.path.expanduser(model_params.model),
        'weights': os.path.expanduser(model_params.weights)
    }
    fp32_model = load_model(model_config, target_device='CPU')

    return model, fp32_model


def get_synthetic_dataset(model, compress_algorithms, dataset_config, tmp_path):
    """
    Returns a synthetic dataset to test
    """
    engine_config = get_engine_configs()

    data_loader = SyntheticDataLoader(dataset_config)
    label_collector = LabelCollector()
    engine = IEEngine(engine_config, data_loader=data_loader, metric=label_collector)
    pipeline = create_pipeline(compress_algorithms, engine)
    pipeline.evaluate(model)
    data_loader._set_annotations(label_collector.get_annotations())

    # Generate a synthetic dataset to tmp_path
    path_images = os.path.join(tmp_path, 'synthetic_data', 'images')
    path_annotation = os.path.join(tmp_path, 'synthetic_data', 'val.txt')

    if not os.path.exists(path_images):
        os.makedirs(path_images)

    # Store synthetic images
    img2annotation = {}
    for i in range(len(data_loader)):
        img, annotation = data_loader[i]
        img_path = os.path.join(path_images, f'{i}.jpg')
        cv2.imwrite(img_path, img)
        img2annotation[img_path] = annotation

    # Store synthetic annotation
    val_data = [f'{img_path} {annotation}' for img_path, annotation in img2annotation.items()]
    with open(path_annotation, 'w') as f_anno:
        f_anno.write('\n'.join(val_data))

    # Create a dataset_info for the synthetic dataset
    dataset_info = [{
        'name': 'synthetic_dataset',
        'annotation_conversion': {
            'converter': 'imagenet',
            'annotation_file': PosixPath(path_annotation)
        },
        'data_source': PosixPath(path_images),
        'metrics': [
            {
                'name': 'accuracy@top1',
                'type': 'accuracy',
                'top_k': 1,
                'reference': 0.0
            }
        ],
        'preprocessing': [],
        '_command_line_mapping': {'annotation_file': None}
    }]

    return data_loader, dataset_info


def optimize_model(fp32_model, data_loader, compress_algorithms):
    engine_config = get_engine_configs()
    metric = Accuracy(top_k=1)
    engine = IEEngine(engine_config, data_loader=data_loader, metric=metric)
    pipeline = create_pipeline(compress_algorithms, engine)
    compressed_model = pipeline.run(fp32_model)
    return compressed_model


def evaluate_with_synthetic_data(model, model_, model_name, algorithm, preset, tmp_path, dataset_info):
    """
    Evaluates a model with the dataset and returns the metric values
    """
    paths = save_model(model_, tmp_path.as_posix(), model_name)

    algorithm_config = make_algo_config(algorithm, preset)
    engine_config = get_engine_config(model_name)
    config = merge_configs(model.model_params, engine_config, algorithm_config)
    config.engine = get_engine_config(model_name)
    from tools.evaluate import evaluate
    for model in config.engine['models']:
        model['datasets'] = dataset_info

    metrics = evaluate(config=config, subset=range(100), paths=paths)

    return metrics


def test_sample_compression(_params, tmp_path, models):
    model_name, model_framework, algorithm, preset, expected_accuracy, dataset_config, optimize_params = _params

    compress_algorithms = [
        {
            'name': algorithm,
            'params': optimize_params
        }
    ]

    model, fp32_model = get_model(models, model_name, model_framework, tmp_path)
    data_loader, dataset_info = get_synthetic_dataset(fp32_model, compress_algorithms, dataset_config, tmp_path)
    compress_model = optimize_model(fp32_model, data_loader, compress_algorithms)
    metrics = evaluate_with_synthetic_data(model, compress_model, model_name, algorithm, preset, tmp_path, dataset_info)

    metrics = OrderedDict([(metric.name, np.mean(metric.evaluated_value))
                           for metric in metrics])

    for metric_name, metric_val in metrics.items():
        print('{}: {:.4f}'.format(metric_name, metric_val))
        if metric_name == 'accuracy@top1':
            assert {metric_name: metric_val} == pytest.approx(expected_accuracy, abs=0.2)

