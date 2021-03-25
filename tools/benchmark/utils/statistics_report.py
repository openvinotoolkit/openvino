# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from enum import Enum

from .logging import logger

## statistics reports types
noCntReport = 'no_counters'
averageCntReport = 'average_counters'
detailedCntReport = 'detailed_counters'

## Responsible for collecting of statistics and dumping to .csv file
class StatisticsReport:
    class Config():
        def __init__(self, report_type, report_folder):
            self.report_type = report_type
            self.report_folder = report_folder

    class Category(Enum):
        COMMAND_LINE_PARAMETERS = 0,
        RUNTIME_CONFIG = 1,
        EXECUTION_RESULTS = 2

    def __init__(self, config):
        self.config = config
        self.parameters = {}
        self.csv_separator = ';'

    def add_parameters(self, category, parameters):
        if category not in self.parameters.keys():
            self.parameters[category] = parameters
        else:
            self.parameters[category].extend(parameters)

    def dump(self):
        def dump_parameters(f, parameters):
            for k, v in parameters:
                f.write(f'{k}{self.csv_separator}{v}\n')

        with open(os.path.join(self.config.report_folder, 'benchmark_report.csv'), 'w') as f:
            if self.Category.COMMAND_LINE_PARAMETERS in self.parameters.keys():
                f.write('Command line parameters\n')
                dump_parameters(f, self.parameters[self.Category.COMMAND_LINE_PARAMETERS])
                f.write('\n')

            if self.Category.RUNTIME_CONFIG in self.parameters.keys():
                f.write('Configuration setup\n')
                dump_parameters(f, self.parameters[self.Category.RUNTIME_CONFIG])
                f.write('\n')

            if self.Category.EXECUTION_RESULTS in self.parameters.keys():
                f.write('Execution results\n')
                dump_parameters(f, self.parameters[self.Category.EXECUTION_RESULTS])
                f.write('\n')

            logger.info(f"Statistics report is stored to {f.name}")

    def dump_performance_counters_request(self, f, perf_counts):
        total = 0
        total_cpu = 0
        f.write(self.csv_separator.join(['layerName', 'execStatus', 'layerType', 'execType', 'realTime (ms)', 'cpuTime (ms)\n']))
        for k, v in sorted(perf_counts.items(), key=lambda x: x[1]['execution_index']):
            f.write(self.csv_separator.join([k, v['status'], v['layer_type'], v['exec_type'], str(v['real_time']/1000.0), str(v['cpu_time']/1000.0)]))
            f.write('\n')
            total += v['real_time']
            total_cpu += v['cpu_time']
        f.write(self.csv_separator.join(['Total','','','',str(total/1000.0),str(total_cpu/1000.0)]))
        f.write('\n\n')

    def dump_performance_counters(self, perf_counts):
        if self.config.report_type == '' or self.config.report_type == noCntReport:
            logger.info("Statistics collecting for performance counters was not requested. No reports are dumped.")
            return

        if not perf_counts:
            logger.info('Performance counters are empty. No reports are dumped.')
            return

        filename = os.path.join(self.config.report_folder, f'benchmark_{self.config.report_type}_report.csv')
        with open(filename, 'w') as f:
            if self.config.report_type == detailedCntReport:
                for pc in perf_counts:
                    self.dump_performance_counters_request(f, pc)
            elif self.config.report_type == averageCntReport:
                def get_average_performance_counters(perf_counts):
                    performance_counters_avg = {}
                    ## iterate over each processed infer request and handle its PM data
                    for i in range(0, len(perf_counts)):
                        ## iterate over each layer from sorted vector and add required PM data to the per-layer maps
                        for k in perf_counts[0].keys():
                            if k not in performance_counters_avg.keys():
                                performance_counters_avg[k] = perf_counts[i][k]
                            else:
                                performance_counters_avg[k]['real_time'] += perf_counts[i][k]['real_time']
                                performance_counters_avg[k]['cpu_time'] += perf_counts[i][k]['cpu_time']
                    for _, v in performance_counters_avg.items():
                        v['real_time'] /= len(perf_counts)
                        v['cpu_time'] /= len(perf_counts)
                    return performance_counters_avg
                self.dump_performance_counters_request(f, get_average_performance_counters(perf_counts))
            else:
                raise Exception('PM data can only be collected for average or detailed report types')

            logger.info(f'Performance counters report is stored to {filename}')
