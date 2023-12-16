# Performance counters

Subgraph in the the snippets could be very large. Sometimes developers are interested the detailed performance number of part of the subgraph. This feature help to do it by inserting perf_count_begin and perf_count_end operations around the sequence of expression in LIR, which developers would like to benchmark. There is an example to insert between last parameter and first result with a transformation(src/common/snippets/src/lowered/pass/insert_perf_count.cpp). Developers could adjust it to benchmark interested sequence.

There are two perf count modes.
 - `Chrono` : perf count with chrono call. This is a universal method, and support multi-threads case to print perf count data for each thread.
 - `BackendSpecific` : perf count provided by backend. This is for device specific requirement. For example, in sake of more light overhead and more accurate result, x86 or x86-64 CPU specific mode via reading RDTSC register is implemented. At current this CPU BackendSpecific mode only support single thread.
 One can select prefered mode by setting perf_count_mode default value in [snippets Config](../../include/snippets/lowered/linear_ir.hpp)
