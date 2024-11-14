# Debug capabilities
Debug capabilities are the set of useful debug features, controlled by environment variables.

They can be activated at runtime and might be used for analyzing issues, getting more context, comparing execution results, etc.

Use the following cmake option to enable debug capabilities:

`-DENABLE_DEBUG_CAPS=ON`

* [Verbose mode](verbose.md)  
  `OV_CPU_VERBOSE=1`
* [Blob dumping](blob_dumping.md)  
  `OV_CPU_BLOB_DUMP_NODE_NAME="*" OV_CPU_BLOB_DUMP_DIR=blob_dump`
* [Graph serialization](graph_serialization.md)  
  `OV_CPU_EXEC_GRAPH_PATH=graph.xml`
  `OV_CPU_DUMP_IR="transformations dir=path/dumpdir formats=svg,xml,dot"`
* [Graph transformation disabling](feature_disabling.md#graph-transformations)  
  `OV_CPU_DISABLE="transformations=common,preLpt,lpt,postLpt,snippets,specific"`
* [Logging](logging.md)  
  `OV_CPU_DEBUG_LOG=-`
* [Inference Precision](infer_prc.md)  
  `OV_CPU_INFER_PRC_POS_PATTERN="^FullyConnected@"`
* Performance summary  
  `OV_CPU_SUMMARY_PERF=1`  
  Set the environment variable to display performance summary at the time when model is being destructed.  
  Internal performance counter will be enabled automatically. 
