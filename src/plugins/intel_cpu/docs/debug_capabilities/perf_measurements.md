# Performance measurements

Performance counters are controlled by plugin config option **PERF_COUNT** and forcibly enabled in [Verbose mode](verbose.md). If it is enabled *total* performance counters are aggregated by default and could be retrieved via API (for example, *-pc* option in *benchmark_app*).

## Aggregate performance counters

If performance counters are enabled, aggregation of additional performance counters is controlled by environment variable **OV_CPU_PERF_TABLES_PATH**:
```sh
    OV_CPU_PERF_TABLES_PATH=<path/prefix> binary ...
```

\<prefix\> is optional, so slash have to be stated at the end of path if no prefix:
```sh
    OV_CPU_PERF_TABLES_PATH="/path/to/dumpDir/" binary ...
    OV_CPU_PERF_TABLES_PATH="/path/to/dumpDir/table_prefix_" binary ...
```

CSV tables with aggregate performance counters are dumped upon *ExecNetwork* destruction:
* \<prefix\>perf_modelInputs.csv\
Map of model input shapes to indexes.
* \<prefix\>perf_raw_nodes.csv\
Aggregate performance counters without processing.
* \<prefix\>perf_0_\<0_model_input_shape\>\_nodes.csv ... \<prefix\>perf_N_\<N_model_input_shape\>\_nodes.csv\
Aggregate performance counters per node for corresponding model inputs.
* \<prefix\>perf_0_\<0_model_input_shape\>\_nodeTypes.csv ... \<prefix\>perf_N_\<N_model_input_shape\>\_nodeTypes.csv\
Aggregate performance counters per node type for corresponding model inputs.
* \<prefix\>perf_all_nodeTypes.csv\
Aggregate performance counters per node type for all model inputs.\
This table is absent in case of single model input (static case) since it is the same as \<prefix\>perf_0_\<0_model_input_shape\>_nodeTypes.csv

\<model_input_shape\> in table name is limited by 100 characters, so complete shape is additionally stated at the end of corresponding table and in \<prefix\>perf_modelInputs.csv.
