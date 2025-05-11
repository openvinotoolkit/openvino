# Average counters

To enable collection of per-node average counters the following environment variable should be used:

```sh
    OV_CPU_AVERAGE_COUNTERS=<filename> binary ...
```

The output table has the same format as:

```sh
    benchmark_app --report_type average_counters
```

The avarage counters can be aggregated using:

* [aggregate-average-counters.py](../../tools/aggregate-average-counters/aggregate-average-counters.py)
