# Debug capabilities
Debug capabilities are the set of useful debug features, controlled by environment variables.

They can be activated at runtime and might be used for analyzing issues, getting more context, comparing execution results, etc.

Use the following cmake option to enable debug capabilities:

`-DENABLE_DEBUG_CAPS=ON`

* [Verbose mode](verbose.md)
  When to use: need to understand execution flow — traces node execution with types, shapes, algorithms, and kernels.
  Example: `OV_CPU_VERBOSE=1`

* [Blob dumping](blob_dumping.md)
  When to use: wrong results / accuracy issues — dump layer I/O tensors and compare against a reference.
  Warning: raw blobs can have very big sizes
  Example: `OV_CPU_BLOB_DUMP_NODE_NAME="*" OV_CPU_BLOB_DUMP_DIR=blob_dump`

* [Execution graph serialization](exec_graph_serialization.md)
  When to use: need to have full execution flow picture, i.e. complete graph structure, how nodes are related, who are the producers / consumers of a node.
  Example: `OV_CPU_EXEC_GRAPH_PATH=graph.xml`

* [IR serialization](ir_serialization.md)
  When to use: unexpected graph structure — serialize IR at each transformation stage to see what changed.
  Example: `OV_CPU_DUMP_IR="transformations dir=path/dumpdir formats=svg,xml,dot"`

* [Graph transformation disabling](feature_disabling.md#graph-transformations)
  When to use: crash or wrong results suspected in a transformation — disable groups to isolate the fault.
  Example: `OV_CPU_DISABLE="transformations=common,preLpt,lpt,postLpt,snippets,specific"`

* [Logging](logging.md)
  When to use: need to debug a specific code path — enables `[ DEBUG ]` logs with source location, supports breakpoint traps. Logs can be filtered and used as breakpoints with `OV_CPU_DEBUG_LOG_BRK`.
  Example: `OV_CPU_DEBUG_LOG=-`

* [Inference Precision](infer_prc.md)
  When to use: precision-related accuracy issues — override inference precision for specific nodes.
  Example: `OV_CPU_INFER_PRC_POS_PATTERN="^FullyConnected@"`

* [Average counters](average_counters.md)
  When to use:
  - performance analysis across multiple inferences — collects averaged execution counters.
  - list of executed nodes, their types and primitive types
  Example: `OV_CPU_AVERAGE_COUNTERS=filename`

* Performance summary
  When to use: slow inference — displays per-node timing summary when the model is destructed.
  Example: `OV_CPU_SUMMARY_PERF=1`

* Memory statistics
  When to use:
  - high memory usage or just memory profiling — dumps memory usage statistics per compiled model.
  Example: `OV_CPU_MEMORY_STATISTICS_PATH=<file_path>.csv`
