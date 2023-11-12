# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import os.path
import shutil
import pytest
from py.xml import html

from models_hub_common.constants import performance_results_path
from models_hub_common.constants import wget_cache_dir
from models_hub_common.utils import get_params


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


def pytest_sessionstart(session):
    objs_to_remove = []
    if performance_results_path and os.path.exists(performance_results_path):
        objs_to_remove.append(performance_results_path)
    for file_name in os.listdir(wget_cache_dir):
        file_path = os.path.join(wget_cache_dir, file_name)
        objs_to_remove.append(file_path)
    for file_path in objs_to_remove:
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            pass


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == 'teardown':
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


@pytest.mark.optionalhook
def pytest_html_results_table_row(report, cells):
    cells.insert(2, html.td(report._results.status))
    cells.insert(3, html.td(report._results.converted_infer_time))
    cells.insert(4, html.td(report._results.converted_model_time_variance))
    cells.insert(5, html.td(report._results.read_model_infer_time))
    cells.insert(6, html.td(report._results.read_model_infer_time_variance))
    cells.insert(7, html.td(report._results.infer_time_ratio))
