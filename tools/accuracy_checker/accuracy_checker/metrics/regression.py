"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings
import math
import numpy as np

from ..representation import (
    RegressionAnnotation,
    RegressionPrediction,
    FacialLandmarksAnnotation,
    FacialLandmarksPrediction,
    SuperResolutionAnnotation,
    SuperResolutionPrediction,
    GazeVectorAnnotation,
    GazeVectorPrediction
)

from .metric import PerImageEvaluationMetric, BaseMetricConfig
from ..config import BaseField, NumberField, BoolField, ConfigError
from ..utils import string_to_tuple, finalize_metric_result


class BaseRegressionMetric(PerImageEvaluationMetric):
    annotation_types = (RegressionAnnotation, )
    prediction_types = (RegressionPrediction, )

    def __init__(self, value_differ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_differ = value_differ

    def configure(self):
        self.meta.update({'names': ['mean', 'std'], 'scale': 1, 'postfix': ' ', 'calculate_mean': False})
        self.magnitude = []

    def update(self, annotation, prediction):
        self.magnitude.append(self.value_differ(annotation.value, prediction.value))

    def evaluate(self, annotations, predictions):
        return np.mean(self.magnitude), np.std(self.magnitude)


class BaseIntervalRegressionMetricConfig(BaseMetricConfig):
    intervals = BaseField(optional=True)
    start = NumberField(optional=True)
    end = NumberField(optional=True)
    step = NumberField(optional=True)
    ignore_values_not_in_interval = BoolField(optional=True)


class BaseRegressionOnIntervals(PerImageEvaluationMetric):
    annotation_types = (RegressionAnnotation, )
    prediction_types = (RegressionPrediction, )

    def __init__(self, value_differ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_differ = value_differ

    def validate_config(self):
        validator = BaseIntervalRegressionMetricConfig(
            'regression_on_intervals_config',
            on_extra_argument=BaseIntervalRegressionMetricConfig.ERROR_ON_EXTRA_ARGUMENT
        )
        validator.validate(self.config)

    def configure(self):
        self.meta.update({'scale': 1, 'postfix': ' ', 'calculate_mean': False})
        self.ignore_out_of_range = self.config.get('ignore_values_not_in_interval', True)

        self.intervals = self.config.get('intervals')
        if not self.intervals:
            stop = self.config.get('end')
            if not stop:
                raise ConfigError('intervals or start-step-end of interval should be specified for metric')

            start = self.config.get('start', 0.0)
            step = self.config.get('step', 1.0)
            self.intervals = np.arange(start, stop + step, step)

        if not isinstance(self.intervals, (list, np.ndarray)):
            self.intervals = string_to_tuple(self.intervals)

        self.intervals = np.unique(self.intervals)
        self.magnitude = [[] for _ in range(len(self.intervals) + 1)]

        self.meta['names'] = ([])
        if not self.ignore_out_of_range:
            self.meta['names'] = (['mean: < ' + str(self.intervals[0]), 'std: < ' + str(self.intervals[0])])

        for index in range(len(self.intervals) - 1):
            self.meta['names'].append('mean: <= ' + str(self.intervals[index]) + ' < ' + str(self.intervals[index + 1]))
            self.meta['names'].append('std: <= ' + str(self.intervals[index]) + ' < ' + str(self.intervals[index + 1]))

        if not self.ignore_out_of_range:
            self.meta['names'].append('mean: > ' + str(self.intervals[-1]))
            self.meta['names'].append('std: > ' + str(self.intervals[-1]))

    def update(self, annotation, prediction):
        index = find_interval(annotation.value, self.intervals)
        self.magnitude[index].append(self.value_differ(annotation.value, prediction.value))

    def evaluate(self, annotations, predictions):
        if self.ignore_out_of_range:
            self.magnitude = self.magnitude[1:-1]

        result = [[np.mean(values), np.std(values)] if values else [np.nan, np.nan] for values in self.magnitude]
        result, self.meta['names'] = finalize_metric_result(np.reshape(result, -1), self.meta['names'])

        if not result:
            warnings.warn("No values in given interval")
            result.append(0)

        return result


class MeanAbsoluteError(BaseRegressionMetric):
    __provider__ = 'mae'

    def __init__(self, *args, **kwargs):
        super().__init__(mae_differ, *args, **kwargs)


class MeanSquaredError(BaseRegressionMetric):
    __provider__ = 'mse'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)


class RootMeanSquaredError(BaseRegressionMetric):
    __provider__ = 'rmse'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)

    def evaluate(self, annotations, predictions):
        return np.sqrt(np.mean(self.magnitude)), np.sqrt(np.std(self.magnitude))


class MeanAbsoluteErrorOnInterval(BaseRegressionOnIntervals):
    __provider__ = 'mae_on_interval'

    def __init__(self, *args, **kwargs):
        super().__init__(mae_differ, *args, **kwargs)


class MeanSquaredErrorOnInterval(BaseRegressionOnIntervals):
    __provider__ = 'mse_on_interval'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)


class RootMeanSquaredErrorOnInterval(BaseRegressionOnIntervals):
    __provider__ = 'rmse_on_interval'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)

    def evaluate(self, annotations, predictions):
        if self.ignore_out_of_range:
            self.magnitude = self.magnitude[1:-1]

        result = []
        for values in self.magnitude:
            error = [np.sqrt(np.mean(values)), np.sqrt(np.std(values))] if values else [np.nan, np.nan]
            result.append(error)

        result, self.meta['names'] = finalize_metric_result(np.reshape(result, -1), self.meta['names'])

        if not result:
            warnings.warn("No values in given interval")
            result.append(0)

        return result


class FacialLandmarksPerPointNormedError(PerImageEvaluationMetric):
    __provider__ = 'per_point_normed_error'

    annotation_types = (FacialLandmarksAnnotation, )
    prediction_types = (FacialLandmarksPrediction, )

    def configure(self):
        self.meta.update({'scale': 1, 'postfix': ' ', 'calculate_mean': True, 'data_format': '{:.4f}'})
        self.magnitude = []

    def update(self, annotation, prediction):
        result = point_regression_differ(
            annotation.x_values, annotation.y_values, prediction.x_values, prediction.y_values
        )
        result /= np.maximum(annotation.interocular_distance, np.finfo(np.float64).eps)
        self.magnitude.append(result)

    def evaluate(self, annotations, predictions):
        num_points = np.shape(self.magnitude)[1]
        point_result_name_pattern = 'point_{}_normed_error'
        self.meta['names'] = [point_result_name_pattern.format(point_id) for point_id in range(num_points)]
        per_point_rmse = np.mean(self.magnitude, axis=1)
        per_point_rmse, self.meta['names'] = finalize_metric_result(per_point_rmse, self.meta['names'])

        return per_point_rmse


class NormedErrorMetricConfig(BaseMetricConfig):
    calculate_std = BoolField(optional=True)
    percentile = NumberField(optional=True, floats=False, min_value=0, max_value=100)


class FacialLandmarksNormedError(PerImageEvaluationMetric):
    __provider__ = 'normed_error'

    annotation_types = (FacialLandmarksAnnotation, )
    prediction_types = (FacialLandmarksPrediction, )

    def validate_config(self):
        config_validator = NormedErrorMetricConfig(
            'normed_error_config', NormedErrorMetricConfig.ERROR_ON_EXTRA_ARGUMENT
        )
        config_validator.validate(self.config)

    def configure(self):
        self.calculate_std = self.config.get('calculate_std', False)
        self.percentile = self.config.get('percentile')
        self.meta.update({
            'scale': 1,
            'postfix': ' ',
            'calculate_mean': not self.calculate_std or not self.percentile,
            'data_format': '{:.4f}',
            'names': ['mean']
        })
        self.magnitude = []

    def update(self, annotation, prediction):
        per_point_result = point_regression_differ(
            annotation.x_values, annotation.y_values, prediction.x_values, prediction.y_values
        )
        avg_result = np.sum(per_point_result) / len(per_point_result)
        avg_result /= np.maximum(annotation.interocular_distance, np.finfo(np.float64).eps)
        self.magnitude.append(avg_result)

    def evaluate(self, annotations, predictions):
        result = [np.mean(self.magnitude)]

        if self.calculate_std:
            result.append(np.std(self.magnitude))
            self.meta['names'].append('std')

        if self.percentile:
            sorted_magnitude = np.sort(self.magnitude)
            index = len(self.magnitude) / 100 * self.percentile
            result.append(sorted_magnitude[int(index)])
            self.meta['names'].append('{}th percentile'.format(self.percentile))

        return result


def calculate_distance(x_coords, y_coords, selected_points):
    first_point = [x_coords[selected_points[0]], y_coords[selected_points[0]]]
    second_point = [x_coords[selected_points[1]], y_coords[selected_points[1]]]
    return np.linalg.norm(np.subtract(first_point, second_point))


def mae_differ(annotation_val, prediction_val):
    return np.abs(annotation_val - prediction_val)


def mse_differ(annotation_val, prediction_val):
    return (annotation_val - prediction_val)**2


def find_interval(value, intervals):
    for index, point in enumerate(intervals):
        if value < point:
            return index

    return len(intervals)


def point_regression_differ(annotation_val_x, annotation_val_y, prediction_val_x, prediction_val_y):
    loss = np.subtract(list(zip(annotation_val_x, annotation_val_y)), list(zip(prediction_val_x, prediction_val_y)))
    return np.linalg.norm(loss, 2, axis=1)


class PeakSignalToNoiseRatio(BaseRegressionMetric):
    __provider__ = 'psnr'

    annotation_types = (SuperResolutionAnnotation, )
    prediction_types = (SuperResolutionPrediction, )

    def __init__(self, *args, **kwargs):
        super().__init__(self._psnr_differ, *args, **kwargs)

    def validate_config(self):
        class _PSNRConfig(BaseMetricConfig):
            scale_border = NumberField(optional=True, min_value=0)

        config_validator = _PSNRConfig('psnr', on_extra_argument=_PSNRConfig.ERROR_ON_EXTRA_ARGUMENT)
        config_validator.validate(self.config)

    def configure(self):
        super().configure()
        self.scale_border = self.config.get('scale_border', 4)

    def _psnr_differ(self, annotation_image, prediction_image):
        prediction = np.asarray(prediction_image).astype(np.float)
        ground_truth = np.asarray(annotation_image).astype(np.float)

        height, width = prediction.shape[:2]
        prediction = prediction[
            self.scale_border:height - self.scale_border,
            self.scale_border:width - self.scale_border
        ]
        ground_truth = ground_truth[
            self.scale_border:height - self.scale_border,
            self.scale_border:width - self.scale_border
        ]
        image_difference = (prediction - ground_truth) / 255.  # rgb color space

        r_channel_diff = image_difference[:, :, 0]
        g_channel_diff = image_difference[:, :, 1]
        b_channel_diff = image_difference[:, :, 2]

        channels_diff = (r_channel_diff * 65.738 + g_channel_diff * 129.057 + b_channel_diff * 25.064) / 256

        mse = np.mean(channels_diff ** 2)
        if mse == 0:
            return np.Infinity

        return -10 * math.log10(mse)


def angle_differ(gt_gaze_vector, predicted_gaze_vector):
    return np.arccos(
        gt_gaze_vector.dot(predicted_gaze_vector) / np.linalg.norm(gt_gaze_vector)
        / np.linalg.norm(predicted_gaze_vector)
    ) * 180 / np.pi


class AngleError(BaseRegressionMetric):
    __provider__ = 'angle_error'

    annotation_types = (GazeVectorAnnotation, )
    prediction_types = (GazeVectorPrediction, )

    def __init__(self, *args, **kwargs):
        super().__init__(angle_differ, *args, **kwargs)
