# Performance counters

Subgraph in snippets could be very large. Sometimes developers are interested the detailed performance number of part of the subgraph. This feature help to do it, by inserting a pair of PerfCountBegin and PerfCountEnd operations around a sequence of expression in LIR(linear IR), which developers would like to benchmark. There is an example to insert between last parameter and first result with a [transformation](../../src/lowered/pass/insert_perf_count.cpp). Developers could adjust it to benchmark their interested sequence.

There are two perf count modes.
 - `Chrono` : Perf count via chrono call. This is a universal method, and support multi-threads scenario to print perf count data for each thread.
 - `BackendSpecific` : Perf count provided by backend. This is for device specific requirement. For example, for sake of more light overhead and more accurate result, x86 or x86-64 CPU specific mode via reading RDTSC register is implemented. At current this x86 or x86-64 CPU BackendSpecific mode only support single thread.
 One can select prefered mode by setting `perf_count_mode` default value in [snippets Config](../../include/snippets/utils/debug_caps_config.hpp)
