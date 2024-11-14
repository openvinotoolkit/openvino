# Debug capabilities
Debug capabilities are the set of useful debug features, most of them are controlled by environment variables.

They can be activated at runtime and might be used to analyze issues, locate faulty code, get more execution details, benchmark execution time, etc.

Use the following cmake option to enable snippets debug capabilities:

`-DENABLE_DEBUG_CAPS=ON`

* [Performance counters](perf_count.md)
* [Snippets segfault detector](snippets_segfault_detector.md)
* [LIR passes serialization](LIR_passes_serialization.md)