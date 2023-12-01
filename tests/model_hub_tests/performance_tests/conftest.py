# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
from models_hub_common.utils import get_params
from py.xml import html
from models_hub_common.utils import round_num


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if getattr(item.obj.__self__, 'result', None) is not None:
        results = item.obj.__self__.result
        report._results = results


@pytest.mark.optionalhook
def pytest_html_results_table_header(cells):
    cells.insert(3, html.th('status', class_="sortable"))
    cells.insert(4, html.th('converted model infer time secs'))
    cells.insert(5, html.th('converted model infer time variance'))
    cells.insert(6, html.th('converted model measurement n repeats'))
    cells.insert(7, html.th('read model infer time secs'))
    cells.insert(8, html.th('read model infer time variance'))
    cells.insert(9, html.th('read model measurement n repeats'))
    cells.insert(10, html.th('model infer time ratio converted_model_time/read_model_time'))


@pytest.mark.optionalhook
def pytest_html_results_table_row(report, cells):
    if getattr(report, '_results', None) is None:
        return
    cells.insert(3, html.td(report._results.status))
    cells.insert(4, html.td(round_num(report._results.converted_model_results.infer_mean_time / (10 ** 9))))
    cells.insert(5, html.td(round_num(report._results.converted_model_results.infer_variance)))
    cells.insert(6, html.td(report._results.converted_model_results.infer_n_repeats))
    cells.insert(7, html.td(round_num(report._results.read_model_results.infer_mean_time / (10 ** 9))))
    cells.insert(8, html.td(round_num(report._results.read_model_results.infer_variance)))
    cells.insert(9, html.td(report._results.read_model_results.infer_n_repeats))
    cells.insert(10, html.td(round_num(report._results.infer_time_ratio)))
