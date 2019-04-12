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

EvaluationResult = namedtuple('EvaluationResult', ['evaluated_value', 'reference_value', 'name', 'threshold', 'meta'])


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
        value, reference, name, threshold, meta = evaluation_result
        value = np.mean(value)
        postfix, scale, result_format = get_result_format_parameters(meta, ignore_results_formatting)
        write_scalar_result(
            value, name, reference, threshold, postfix=postfix, scale=scale, result_format=result_format
        )


class VectorPrintPresenter(BasePresenter):
    __provider__ = "print_vector"

    def write_result(self, evaluation_result: EvaluationResult, output_callback=None, ignore_results_formatting=False):
        value, reference, name, threshold, meta = evaluation_result
        if threshold:
            threshold = float(threshold)

        value_names = meta.get('names')
        postfix, scale, result_format = get_result_format_parameters(meta, ignore_results_formatting)
        if np.isscalar(value) or np.size(value) == 1:
            value = [value]

        for index, res in enumerate(value):
            write_scalar_result(
                res, name, reference, threshold,
                value_name=value_names[index] if value_names else None,
                postfix=postfix[index] if not np.isscalar(postfix) else postfix,
                scale=scale[index] if not np.isscalar(scale) else scale,
                result_format=result_format
            )

        if len(value) > 1 and meta.get('calculate_mean', True):
            write_scalar_result(
                np.mean(np.multiply(value, scale)), name, reference, threshold, value_name='mean',
                postfix=postfix[-1] if not np.isscalar(postfix) else postfix, scale=1,
                result_format=result_format
            )


def write_scalar_result(res_value, name, reference, threshold, value_name=None, postfix='%', scale=100,
                        result_format='{:.2f}'):
    display_name = "{}@{}".format(name, value_name) if value_name else name
    display_result = result_format.format(res_value * scale)
    message = '{}: {}{}'.format(display_name, display_result, postfix)

    if reference:
        threshold = threshold or 0

        difference = abs(reference - (res_value * scale))
        if threshold <= difference:
            fail_message = "[FAILED: error = {:.4}]".format(difference)
            message = "{} {}".format(message, color_format(fail_message, Color.FAILED))
        else:
            message = "{} {}".format(message, color_format("[OK]", Color.PASSED))

    print_info(message)


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
