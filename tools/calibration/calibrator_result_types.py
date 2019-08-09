class RawResults:
    def __init__(self, fp32_stats, fp32_latency):
        self.fp32_stats = fp32_stats
        self.fp32_latency = fp32_latency


class LowPrecisionResults:
    def __init__(self):
        self.accuracy = None
        self.latency = 0.0
        self.threshold = 100.0
        self.statistics = None
        self.performance_counters = None
        self.accuracy_drop = None
        self.accuracy_fits_threshold = False
