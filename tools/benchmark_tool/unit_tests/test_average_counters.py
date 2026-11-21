# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from datetime import timedelta

import pytest
from openvino import ProfilingInfo
from openvino.tools.benchmark.utils.statistics_report import (
    CsvStatisticsReport,
    JsonStatisticsReport,
    StatisticsReport,
    averageCntReport,
)


def counter(name, milliseconds):
    info = ProfilingInfo()
    info.node_name, info.node_type, info.exec_type = name, 'Add', 'ref'
    info.status = ProfilingInfo.Status.EXECUTED
    info.real_time = timedelta(milliseconds=milliseconds)
    info.cpu_time = timedelta(milliseconds=milliseconds * 2)
    return info


@pytest.mark.parametrize('report_class', [CsvStatisticsReport, JsonStatisticsReport])
def test_average_counters_keeps_every_layer_without_mutation(tmp_path, report_class):
    counters = [[counter('first', 2), counter('second', 6)],
                [counter('first', 4), counter('second', 8)]]
    original = [[(pi.real_time, pi.cpu_time) for pi in request] for request in counters]
    report = report_class(StatisticsReport.Config(averageCntReport, str(tmp_path)))
    report.dump_performance_counters(counters)
    if report_class is JsonStatisticsReport:
        result = json.loads((tmp_path / 'benchmark_average_counters_report.json').read_text())
        averages = result['avg_performance']
        assert [node['name'] for node in averages['nodes']] == ['first', 'second']
        assert [node['real_time'] for node in averages['nodes']] == ['3.000', '7.000']
        assert all(node['node_type'] == 'Add' and node['exec_type'] == 'ref' for node in averages['nodes'])
        assert averages['total_real_time'] == '10.000'
        assert averages['total_cpu_time'] == '20.000'
    else:
        with (tmp_path / 'benchmark_average_counters_report.csv').open(newline='') as file:
            rows = list(csv.reader(file, delimiter=';'))
        assert rows[1][:1] == ['first']
        assert rows[2][:1] == ['second']
        assert [rows[1][4], rows[2][4]] == ['3.000', '7.000']
    assert [[(pi.real_time, pi.cpu_time) for pi in request] for request in counters] == original
    filename = next(tmp_path.iterdir())
    first_report = filename.read_text()
    report.dump_performance_counters(counters)
    assert filename.read_text() == first_report
    assert [[(pi.real_time, pi.cpu_time) for pi in request] for request in counters] == original


def test_average_json_handles_empty_request_and_layer_order(tmp_path):
    counters = [[counter('first', 6), counter('second', 9)], [],
                [counter('second', 3), counter('first', 0)]]
    report = JsonStatisticsReport(StatisticsReport.Config(averageCntReport, str(tmp_path)))
    report.dump_performance_counters(counters)
    result = json.loads((tmp_path / 'benchmark_average_counters_report.json').read_text())
    nodes = result['avg_performance']['nodes']
    assert [(node['name'], node['real_time']) for node in nodes] == [('first', '2.000'), ('second', '4.000')]
