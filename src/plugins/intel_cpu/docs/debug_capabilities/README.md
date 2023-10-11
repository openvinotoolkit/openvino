# Debug capabilities
Debug capabilities are the set of useful debug features, controlled by environment variables.

They can be activated at runtime and might be used for analyzing issues, getting more context, comparing execution results, etc.

Use the following cmake option to enable debug capabilities:

`-DENABLE_DEBUG_CAPS=ON`

* [Verbose mode](verbose.md)
* [Blob dumping](blob_dumping.md)
* [Graph serialization](graph_serialization.md)
* [Graph transformation disabling](feature_disabling.md#graph-transformations)
* [Logging](logging.md)
* [Inference Precision](infer_prc.md)
* Performance summary
    * set `OV_CPU_SUMMARY_PERF` environment variable to display performance summary at the time when model is being destructed.
    * Internal performance counter will be enabled automatically. 
