import yaml
import copy
import statistics


class StatisticsParser:
    executor = None

    def raise_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    def _set_executor(self, stats: list):
        if self.executor:
            return
        else:
            for executor in [MemoryStatisticsParser]:
                try:
                    executor().parse_stats(stats)
                    self.executor = executor()
                    break
                except:
                    self.executor = None
            if not self.executor:
                self.raise_not_implemented()

    def parse_stats(self, stats: list):
        self._set_executor(stats)
        getattr(self.executor, "parse_stats", self.raise_not_implemented)(stats)

    def append_stats(self, stats: list):
        self._set_executor(stats)
        getattr(self.executor, "append_stats", self.raise_not_implemented)(stats)

    def aggregate_stats(self):
        getattr(self.executor, "aggregate_stats", self.raise_not_implemented)()


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
        self.aggregated_stats = {step_name: {vm_metric: {
            "avg": statistics.mean(vm_values_list),
            "stdev": statistics.stdev(vm_values_list) if len(vm_values_list) > 1 else 0}
                    for vm_metric, vm_values_list in vm_values.items()}
            for step_name, vm_values in self.combined_stats.items()}


stats_parser = StatisticsParser()
for run_iter in range(3):
    with open(rf"C:\Applications\OpenVINO\tmp\raw{run_iter}.yml", "r") as file:
        raw_data = list(yaml.load_all(file, Loader=yaml.SafeLoader))

    stats_parser.append_stats(raw_data[0])

stats_parser.aggregate_stats()
