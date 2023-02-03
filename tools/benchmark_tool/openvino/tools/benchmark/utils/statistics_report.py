# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
import os
import sys
from enum import Enum
import csv
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

    def dump_performance_counters_request(self, f, prof_info):
        total = timedelta()
        total_cpu = timedelta()
        f.write(self.csv_separator.join(['layerName', 'execStatus', 'layerType', 'execType', 'realTime (ms)', 'cpuTime (ms)\n']))
        for pi in prof_info:
            f.write(self.csv_separator.join([pi.node_name, str(pi.status), pi.node_type, pi.exec_type, 
                f"{pi.real_time / timedelta(milliseconds=1):.3f}", 
                f"{pi.cpu_time / timedelta(milliseconds=1):.3f}"]))
            f.write('\n')
            total += pi.real_time
            total_cpu += pi.cpu_time
        f.write(self.csv_separator.join(['Total', '', '', '',
            f"{total / timedelta(milliseconds=1):.3f}",
            f"{total_cpu / timedelta(milliseconds=1):.3f}"]))
        f.write('\n\n')

    def dump_performance_counters_sorted(self, prof_sorted_info):
        """Save sorted performance counters into csv file.
        """
        filename = os.path.join(self.config.report_folder, f'benchmark_sorted_report.csv')
        total = 0
        total_cpu = 0
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['layerName', 'execStatus', 'layerType', 'execType', 'realTime (ms)', 'cpuTime (ms)' , 'proportion (%)\n'])
            for tmp_prof in prof_sorted_info:
                writer.writerow([tmp_prof[0], str(tmp_prof[1]), 
                                    tmp_prof[2], tmp_prof[6], 
                                    f"{tmp_prof[3] / 1000:.3f}", # Divide by 1000
                                    f"{tmp_prof[4] / 1000:.3f}",
                                    str("%.2f"%(tmp_prof[5]*100))+"%"])
                total += tmp_prof[3]
                total_cpu += tmp_prof[4]
            f.write('\n')
            writer.writerow(["Total time: %.2f milliseconds"%(total / 1000)])
            writer.writerow(["Total CPU time: %.2f milliseconds"%(total_cpu / 1000)])
            f.write('\n\n')            
        logger.info(f'Sorted performance counters report is stored to {filename}')

    def dump_performance_counters(self, prof_info_list):
        if self.config.report_type == '' or self.config.report_type == noCntReport:
            logger.info("Statistics collecting for performance counters was not requested. No reports are dumped.")
            return

        if not prof_info_list:
            logger.info('Performance counters are empty. No reports are dumped.')
            return

        filename = os.path.join(self.config.report_folder, f'benchmark_{self.config.report_type}_report.csv')
        with open(filename, 'w') as f:
            if self.config.report_type == detailedCntReport:
                for prof_info in prof_info_list:
                    self.dump_performance_counters_request(f, prof_info)
            elif self.config.report_type == averageCntReport:
                def get_average_performance_counters(prof_info_list):
                    performance_counters_avg = []
                    ## iterate over each processed infer request and handle its PM data
                    for prof_info in prof_info_list:
                        for pi in prof_info:
                            item = next((x for x in performance_counters_avg if x.node_name == pi.node_name), None)
                            if item:
                                item.real_time += pi.real_time
                                item.cpu_time += pi.cpu_time
                            else:
                                performance_counters_avg.append(pi)

                    for pi in performance_counters_avg:
                        pi.real_time /= len(prof_info_list)
                        pi.cpu_time /= len(prof_info_list)
                    return performance_counters_avg
                self.dump_performance_counters_request(f, get_average_performance_counters(prof_info_list))
            else:
                raise Exception('PM data can only be collected for average or detailed report types')

            logger.info(f'Performance counters report is stored to {filename}')
