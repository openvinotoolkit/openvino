# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
from datetime import timedelta

from openvino import ProfilingInfo
from openvino.tools.benchmark.utils.statistics_report import (
    CsvStatisticsReport,
    StatisticsReport,
    detailedCntReport,
)


def test_csv_summary_quotes_delimiters_quotes_and_newlines(tmp_path):
    report = CsvStatisticsReport(StatisticsReport.Config('', str(tmp_path)))
    parameters = [('model', 'folder;name/model.xml'), ('device', 'GPU "test"'),
                  ('name', 'first\nsecond'), ('carriage_return', 'first\rsecond')]
    report.add_parameters(StatisticsReport.Category.COMMAND_LINE_PARAMETERS, parameters)
    report.dump()
    with (tmp_path / 'benchmark_report.csv').open(newline='') as file:
        rows = list(csv.reader(file, delimiter=';'))
    assert rows[1:5] == [list(pair) for pair in parameters]


def test_csv_counters_quote_node_names(tmp_path):
    report = CsvStatisticsReport(StatisticsReport.Config(detailedCntReport, str(tmp_path)))
    info = ProfilingInfo()
    info.node_name, info.node_type, info.exec_type = 'layer;"first"\nsecond', 'Add', 'ref'
    info.real_time = info.cpu_time = timedelta(milliseconds=2)
    report.dump_performance_counters([[info]])
    with (tmp_path / 'benchmark_detailed_counters_report.csv').open(newline='') as file:
        rows = list(csv.reader(file, delimiter=';'))
    assert rows[1][0] == info.node_name
    assert len(rows[1]) == 6
    assert rows[1][4:] == ['2.000', '2.000']
