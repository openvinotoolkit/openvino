# OpenVINO CPU plugin
Development documentation of OpenVINO CPU plugin

## Compilation options
See [Compilation options](compilation_options.md)

## Debug capabilities
Use the following cmake option to enable debug capabilities:

`-DENABLE_DEBUG_CAPS=ON`

* [Verbose mode](verbose.md)
* [Blob dumping](blob_dumping.md)
* [Graph serialization](graph_serialization.md)
* [Graph transformation disabling](feature_disabling.md#graph-transformations)
* [Logging](logging.md)
* Performance summary
    * set `OV_CPU_SUMMARY_PERF` environment variable to display performance summary at the time when model is being destructed.
    * Internal performance counter will be enabled automatically. 
