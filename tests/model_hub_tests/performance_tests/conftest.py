# Copyright (C) 2018-2024 Intel Corporation
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
    cells.insert(3, html.th('Status', class_="sortable"))
    cells.insert(4, html.th('convert_model Pipeline Inference Time, sec.'))
    cells.insert(5, html.th('read_model Pipeline Inference Time, sec.'))
    cells.insert(6, html.th('Inference Time Ratio (convert_model vs. read_model)'))


@pytest.mark.optionalhook
def pytest_html_results_table_row(report, cells):
    if getattr(report, '_results', None) is None:
        return
    cells.insert(3, html.td(str(report._results.status)[7:]))
    cells.insert(4, html.td(round_num(report._results.converted_model_results.infer_mean_time)))
    cells.insert(5, html.td(round_num(report._results.read_model_results.infer_mean_time)))
    cells.insert(6, html.td(round_num(report._results.infer_time_ratio)))
