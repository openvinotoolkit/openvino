# CPU Plugin debug capabilities

The page describes list of useful debug features, controlled by environment variables.

They can be activated at runtime and might be used for analyzing issues, getting more context, comparing execution results, etc.

To have CPU debug capabilities available at runtime the following CMake option should be used when building the plugin:
* `ENABLE_DEBUG_CAPS`. Default is `OFF`

The following debug capabilities are available with the latest openvino:

- [Verbose mode](../src/docs/verbose.md)
- [Blob dumping](../src/docs/blob_dumping.md)
- [Graph serialization](../src/docs/graph_serialization.md)