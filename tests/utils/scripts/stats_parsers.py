import statistics
import copy

import numpy as np

# Define a range to cut outliers which are < Q1 − IQR_CUTOFF * IQR, and > Q3 + IQR_CUTOFF * IQR
# https://en.wikipedia.org/wiki/Interquartile_range
IQR_CUTOFF = 1.5


class StatisticsParser:
    executor = None

    def raise_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    def _set_executor(self, stats: list):
        if self.executor:
            return
        else:
            for executor in [TimeStatisticsParser, MemoryStatisticsParser]:
                try:
                    executor().parse_stats(stats)
                    self.executor = executor()
                    break
                except:
                    self.executor = None
            if not self.executor:
                self.raise_not_implemented()

    def parse_stats(self, stats):
        self._set_executor(stats)
        getattr(self.executor, "parse_stats", self.raise_not_implemented)(stats)

    def append_stats(self, stats):
        self._set_executor(stats)
        getattr(self.executor, "append_stats", self.raise_not_implemented)(stats)

    def aggregate_stats(self):
        getattr(self.executor, "aggregate_stats", self.raise_not_implemented)()

    def filter_stats(self):
        getattr(self.executor, "filter_stats", self.raise_not_implemented)()


class TimeStatisticsParser:
    def __init__(self):
        self.last_stats = {}
        self.combined_stats = {}
        self.aggregated_stats = {}
        self.filtered_stats = {}

    def parse_stats(self, stats: list):
        """Parse raw statistics from nested list to flatten dict"""
        for element in stats:
            if isinstance(element, (int, float)):
                for k, v in self.last_stats.items():
                    if v is None:
                        self.last_stats.update({k: [element]})
            else:
                for k, v in element.items():
                    if len(v) == 1:
                        self.last_stats.update({k: v})
                    else:
                        self.last_stats.update({k: None})
                        self.parse_stats(v)

    def append_stats(self, stats: list):
        self.parse_stats(stats)
        if not self.combined_stats:
            self.combined_stats = copy.deepcopy(self.last_stats)
            return
        for step_name, duration in self.last_stats.items():
            self.combined_stats[step_name].extend(duration)

    @staticmethod
    def calculate_iqr(stats: list):
        """IQR is calculated as the difference between the 3th and the 1th quantile of the data."""
        q1 = np.quantile(stats, 0.25)
        q3 = np.quantile(stats, 0.75)
        iqr = q3 - q1
        return iqr, q1, q3

    def filter_stats(self):
        """Identify and remove outliers from statistical data."""
        for step_name, results in self.combined_stats.items():
            iqr, q1, q3 = self.calculate_iqr(results)
            cut_off = iqr * IQR_CUTOFF
            upd_time_results = [x for x in results if (q1 - cut_off < x < q3 + cut_off)]
            self.filtered_stats.update({step_name: upd_time_results if upd_time_results else results})

    def aggregate_stats(self):
        """Aggregate provided statistics"""
        self.aggregated_stats = {step_name: {"avg": statistics.mean(duration_list),
                                             "stdev": statistics.stdev(duration_list) if len(duration_list) > 1 else 0}
                                 for step_name, duration_list in self.filtered_stats.items()}


class MemoryStatisticsParser:
    def __init__(self):
        self.last_stats = {}
        self.combined_stats = {}
        self.aggregated_stats = {}

    def parse_stats(self, stats: dict):
        """Parse statistics to dict"""
        for k, v in stats.items():
            if k not in self.last_stats.keys():
                self.last_stats.update({k: {}})
            if isinstance(v, list):
                for element in v:
                    for metric, value in element.items():
                        self.last_stats[k].update({metric: [value]})

    def append_stats(self, stats: dict):
        self.parse_stats(stats)
        if not self.combined_stats:
            self.combined_stats = copy.deepcopy(self.last_stats)
            return
        for step_name, vm_values in self.last_stats.items():
            for vm_metric, vm_value in vm_values.items():
                self.combined_stats[step_name][vm_metric].extend(vm_value)

    def aggregate_stats(self):
        self.aggregated_stats = {step_name: {vm_metric: {"avg": statistics.mean(vm_values_list),
                                                         "stdev": statistics.stdev(vm_values_list)
                                                         if len(vm_values_list) > 1 else 0}
                                             for vm_metric, vm_values_list in vm_values.items()}
                                 for step_name, vm_values in self.combined_stats.items()}

    def filter_stats(self):
        pass
