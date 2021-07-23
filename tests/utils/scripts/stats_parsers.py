import statistics


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class StatisticsParser(metaclass=Singleton):
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

    def parse_stats(self, stats: list):
        self._set_executor(stats)
        getattr(self.executor, "parse_stats", self.raise_not_implemented)(stats)

    def append_stats(self, stats: list):
        self._set_executor(stats)
        getattr(self.executor, "append_stats", self.raise_not_implemented)(stats)

    def aggregate_stats(self):
        getattr(self.executor, "aggregate_stats", self.raise_not_implemented)()


class TimeStatisticsParser:

    def __init__(self):
        self.last_stats = {}
        self.combined_stats = {}
        self.aggregated_stats = {}

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
            self.combined_stats = self.last_stats.copy()
            return
        for step_name, duration in self.last_stats.items():
            self.combined_stats[step_name].extend(duration)

    def aggregate_stats(self):
        """Aggregate provided statistics"""
        self.aggregated_stats = {step_name: {"avg": statistics.mean(duration_list),
                                             "stdev": statistics.stdev(duration_list) if len(duration_list) > 1 else 0}
                                 for step_name, duration_list in self.last_stats.items()}


class MemoryStatisticsParser:

    def __init__(self):
        self.last_stats = {}
        self.combined_stats = {}
        self.aggregated_stats = {}

    def parse_stats(self, stats: list, key: str = ""):
        """Parse raw statistics from nested list to flatten dict"""
        for element in stats:
            for k, v in element.items():
                if isinstance(v, (int, float)):
                    self.last_stats[key].update({k: [v]})
                else:
                    self.last_stats.update({k: {}})
                    self.parse_stats(v, k)

    def append_stats(self, stats: list):
        self.parse_stats(stats)
        if not self.combined_stats:
            self.combined_stats = self.last_stats.copy()
            return
        for step_name, vm_values in self.last_stats.items():
            for vm_metric, vm_value in vm_values.items():
                self.combined_stats[step_name][vm_metric].extend(vm_value)

    def aggregate_stats(self):
        self.aggregated_stats = {step_name:
                                     {vm_metric:
                                          {"avg": statistics.mean(vm_values_list),
                                           "stdev": statistics.stdev(vm_values_list) if len(vm_values_list) > 1 else 0}
                                      for vm_metric, vm_values_list in vm_values.items()}
                                 for step_name, vm_values in self.last_stats.items()}
