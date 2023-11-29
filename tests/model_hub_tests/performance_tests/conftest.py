# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
from models_hub_common.utils import get_params
from py.xml import html


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == 'teardown' and getattr(item.obj.__self__, 'result', None) is not None:
        results = item.obj.__self__.result
        report._results = results


@pytest.mark.optionalhook
def pytest_html_results_table_header(cells):
    cells.insert(2, html.th('status', class_="sortable"))
    cells.insert(3, html.th('converted model infer time'))
    cells.insert(4, html.th('converted model infer time variance'))
    cells.insert(5, html.th('read model infer time'))
    cells.insert(6, html.th('read model infer time variance'))
    cells.insert(7, html.th('model infer time ratio converted_model_time/read_model_time'))


def round_num(n: float) -> str:
    s = '{:.4E}'.format(n)
    if s.endswith('E+00'):
        return s[:-4]
    return s


@pytest.mark.optionalhook
def pytest_html_results_table_row(report, cells):
    if not getattr(report, '_results', None):
        return
    cells.insert(2, html.td(report._results.status))
    cells.insert(3, html.td(round_num(report._results.converted_infer_time)))
    cells.insert(4, html.td(round_num(report._results.converted_model_time_variance)))
    cells.insert(5, html.td(round_num(report._results.read_model_infer_time)))
    cells.insert(6, html.td(round_num(report._results.read_model_infer_time_variance)))
    cells.insert(7, html.td(round_num(report._results.infer_time_ratio)))
