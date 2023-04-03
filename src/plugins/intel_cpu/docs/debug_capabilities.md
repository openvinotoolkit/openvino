# CPU Plugin Debug Capabilities

The page describes a list of useful debug features, controlled by environment variables.

They can be activated at runtime and might be used for analyzing issues, getting more context, comparing execution results, etc.

To have CPU debug capabilities available at runtime, use the following CMake option when building the plugin:
* `ENABLE_DEBUG_CAPS`. Default is `OFF`

The following debug capabilities are available with the latest OpenVINO:

- [Verbose mode](../src/docs/verbose.md)
- [Blob dumping](../src/docs/blob_dumping.md)
- [Graph serialization](../src/docs/graph_serialization.md)

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)