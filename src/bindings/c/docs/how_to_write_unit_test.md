# How to Write Unit Test for C API

To ensure the accuracy of C API, all interfaces need to implement function level unit test at least. According to the object define, all implemented unit test cases are located in this folder.

The generated binary `ov_capi_test` included the unit cases about all implemented C API interfaces.

If developer wrap new interfaces from OpenVINO C++, you also need to add the unit test case in the correct location.
Here is an example wrap C++ interface to C [wrap core](./how_to_wrap_openvino_interfaces_with_c.md).

Create unit test case for this interface. At first, this interface is for core operation so the location should at [ov_core_test.cpp](../tests/ov_core_test.cpp). Also, the interface has default parameter so need to make unit test case for parameter missing. The final based function level test like:

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/bindings/c/tests/ov_core_test.cpp#L39-L63

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [C API developer guide](../README.md)
 * [C API Reference](https://docs.openvino.ai/2025/api/api_reference.html)

