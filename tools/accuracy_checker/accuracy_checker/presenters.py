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

from collections import namedtuple
from enum import Enum
import numpy as np

from .dependency import ClassProvider
from .logging import print_info

EvaluationResult = namedtuple(
    'EvaluationResult', [
        'evaluated_value', 'reference_value', 'name', 'metric_type', 'threshold', 'meta'
    ]
)


class Color(Enum):
    PASSED = 0
    FAILED = 1


def color_format(s, color=Color.PASSED):
    if color == Color.PASSED:
        return "\x1b[0;32m{}\x1b[0m".format(s)
    return "\x1b[0;31m{}\x1b[0m".format(s)


class BasePresenter(ClassProvider):
    __provider_type__ = "presenter"

    def write_result(self, evaluation_result, output_callback=None, ignore_results_formatting=False):
        raise NotImplementedError


class ScalarPrintPresenter(BasePresenter):
    __provider__ = "print_scalar"

    def write_result(self, evaluation_result: EvaluationResult, output_callback=None, ignore_results_formatting=False):
        value, reference, name, _, threshold, meta = evaluation_result
        value = np.mean(value)
        postfix, scale, result_format = get_result_format_parameters(meta, ignore_results_formatting)
        difference = None
        if reference:
            _, original_scale, _ = get_result_format_parameters(meta, False)
            difference = compare_with_ref(reference, value, original_scale)
        write_scalar_result(
            value, name, threshold, difference, postfix=postfix, scale=scale, result_format=result_format
        )


class VectorPrintPresenter(BasePresenter):
    __provider__ = "print_vector"

    def write_result(self, evaluation_result: EvaluationResult, output_callback=None, ignore_results_formatting=False):
        value, reference, name, _, threshold, meta = evaluation_result
        if threshold:
            threshold = float(threshold)

        value_names = meta.get('names')
        postfix, scale, result_format = get_result_format_parameters(meta, ignore_results_formatting)
        if np.isscalar(value) or np.size(value) == 1:
            if not np.isscalar(value):
                value = value[0]
            difference = None
            if reference:
                _, original_scale, _ = get_result_format_parameters(meta, False)
                difference = compare_with_ref(reference, value, original_scale)
            write_scalar_result(
                value, name, threshold, difference,
                value_name=value_names[0] if value_names else None,
                postfix=postfix[0] if not np.isscalar(postfix) else postfix,
                scale=scale[0] if not np.isscalar(scale) else scale,
                result_format=result_format
            )
            return

        for index, res in enumerate(value):
            cur_postfix = '%'
            if not np.isscalar(postfix):
                if index < len(postfix):
                    cur_postfix = postfix[index]
            else:
                cur_postfix = postfix
            write_scalar_result(
                res, name,
                value_name=value_names[index] if value_names else None,
                postfix=cur_postfix,
                scale=scale[index] if not np.isscalar(scale) else scale,
                result_format=result_format
            )

        if len(value) > 1 and meta.get('calculate_mean', True):
            mean_value = np.mean(np.multiply(value, scale))
            difference = None
            if reference:
                original_scale = get_result_format_parameters(meta, False)[1] if ignore_results_formatting else 1
                difference = compare_with_ref(reference, mean_value, original_scale)
            write_scalar_result(
                mean_value, name, threshold, difference, value_name='mean',
                postfix=postfix[-1] if not np.isscalar(postfix) else postfix, scale=1,
                result_format=result_format
            )


def write_scalar_result(
        res_value, name, threshold=None, diff_with_ref=None, value_name=None,
        postfix='%', scale=100, result_format='{:.2f}'
):
    display_name = "{}@{}".format(name, value_name) if value_name else name
    display_result = result_format.format(res_value * scale)
    message = '{}: {}{}'.format(display_name, display_result, postfix)

    if diff_with_ref:
        threshold = threshold or 0
        if threshold <= diff_with_ref:
            fail_message = "[FAILED: error = {:.4}]".format(diff_with_ref)
            message = "{} {}".format(message, color_format(fail_message, Color.FAILED))
        else:
            message = "{} {}".format(message, color_format("[OK]", Color.PASSED))

    print_info(message)


def compare_with_ref(reference, res_value, scale):
    return abs(reference - (res_value * scale))


class ReturnValuePresenter(BasePresenter):
    __provider__ = "return_value"

    def write_result(self, evaluation_result: EvaluationResult, output_callback=None, ignore_results_formatting=False):
        if output_callback:
            output_callback(evaluation_result)


def get_result_format_parameters(meta, use_default_formatting):
    postfix = ' '
    scale = 1
    result_format = '{}'
    if not use_default_formatting:
        postfix = meta.get('postfix', '%')
        scale = meta.get('scale', 100)
        result_format = meta.get('data_format', '{:.2f}')

    return postfix, scale, result_format
