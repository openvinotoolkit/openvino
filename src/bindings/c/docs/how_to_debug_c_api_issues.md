# How to Debug C API Issues

C API provides exception handling, here is all possible return values of the functions:

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/bindings/c/include/openvino/c/ov_common.h#L68-L96

As known, C API is a wrapping for C++ API, the main issues we had met can be classified into two kinds. Issue from input parameters checking and from C++ call exception.
* for parameter checking issue: return value is -14, please check the input parameters
* for C++ call exception issue: C interface just returns the status value, no more detail info message. If you want details please print it in exception macro.

> **NOTE**: The exception from C interface is not the same as C++ exception, do not use the C status value in C++ debuging.

 ## See also
 * [OpenVINO™ README](../../../../README.md)
 * [C API developer guide](../README.md)
 * [OpenVINO Debug Capabilities](../../../../docs/dev/debug_capabilities.md)
