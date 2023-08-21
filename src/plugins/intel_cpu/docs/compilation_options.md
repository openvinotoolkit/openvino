# Compilation options
The following compilation options can be helpful in developing CPU plugin

* `ENABLE_DEBUG_CAPS=ON`
  See [Debug capabilities](./debug_capabilities/README.md)
* `ENABLE_CPU_SUBSET_TESTS_PATH="relative/path/to/test/file"`
    * Example: `-DENABLE_CPU_SUBSET_TESTS_PATH="single_layer_tests/convolution.cpp subgraph_tests/src/mha.cpp"`
    * Specifies the list of relative paths to functional tests which will be included into the new target `ov_cpu_func_tests_subset`
    * When the option is specified, the target `ov_cpu_func_tests` is excluded from the target `all`
    * Motivation: To reduce overhead of running / debugging the full test scope
* `DENABLE_CPU_SPECIFIC_TARGET_PER_TEST=ON`
    * Generates specific make target for every file from `single_layer_tests` and `subgraph_tests` directories
    * Examples of the generated targets:
        - `ov_cpu_func_slt_convolution` (single_layer_tests/convolution.cpp)
        - `ov_cpu_func_subgraph_mha` (subgraph_tests/src/mha.cpp)
    * All the generated targets are excluded from default target `all`
    * Motivation: To reduce overhead of running / debugging the full test scope
