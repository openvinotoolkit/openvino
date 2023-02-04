# How to support a new OpenVINO opset

Adding a new opset to the OpenVINO IR Frontend is a very frequent task. When you introduce a new operation set, you must also register it in `get_available_opsets()` function.
After that this operation set will automatically supported by IR Frontend.

## See also

 * [OpenVINO IR Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
