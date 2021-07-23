import logging

REFS_FACTOR = 1.2      # 120%


class DummyLogger:
    def error(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


class MetricsComparator:
    executor = None

    def __init__(self, values):
        self.values = values

    def raise_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    def _set_executor(self, values: dict):
        if self.executor:
            return
        else:
            for executor in [TimeMetricsComparator, MemoryMetricsComparator]:
                try:
                    executor(self.values).compare_with(values)
                    self.executor = executor(self.values)
                    break
                except:
                    self.executor = None
            if not self.executor:
                self.raise_not_implemented()

    def compare_with(self, values: dict):
        self._set_executor(values)
        getattr(self.executor, "compare_with", self.raise_not_implemented)(values, log=logging)


class TimeMetricsComparator:

    def __init__(self, values):
        self.status = 0
        self.values = values

    def compare_with(self, reference: dict, log=None):
        """Compare values with provided reference"""
        log = DummyLogger() if not log else log
        for step_name, references in reference.items():
            for metric, reference_val in references.items():
                if self.values[step_name][metric] > reference_val * REFS_FACTOR:
                    log.error("Comparison failed for '{}' step for '{}' metric. Reference: {}. Current values: {}"
                              .format(step_name, metric, reference_val, self.values[step_name][metric]))
                    self.status = 1
                else:
                    log.info("Comparison passed for '{}' step for '{}' metric. Reference: {}. Current values: {}"
                             .format(step_name, metric, reference_val, self.values[step_name][metric]))


class MemoryMetricsComparator:

    def __init__(self, values):
        self.status = 0
        self.values = values

    def compare_with(self, reference: dict, log=None):
        """Compare values with provided reference"""
        log = DummyLogger() if not log else log
        for step_name, vm_records in reference.items():
            for vm_metric, stat_metrics in vm_records.items():
                for stat_metric_name, reference_val in stat_metrics.items():
                    if self.values[step_name][vm_metric][stat_metric_name] > reference_val * REFS_FACTOR:
                        log.error(f"Comparison failed for '{step_name}' step for '{vm_metric}' for"
                                  f" '{stat_metric_name}' metric. Reference: {reference_val}."
                                  f" Current values: {self.values[step_name][vm_metric][stat_metric_name]}")
                        self.status = 1
                    else:
                        log.info(f"Comparison passed for '{step_name}' step for '{vm_metric}' for"
                                 f" '{stat_metric_name}' metric. Reference: {reference_val}."
                                 f" Current values: {self.values[step_name][vm_metric][stat_metric_name]}")
