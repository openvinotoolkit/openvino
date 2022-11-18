# How to Debug C API Issues

C API provides exception handling, here are all possible return values of the functions:

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/bindings/c/include/openvino/c/ov_common.h#L68-L96

There are two main types of possible issues:
* parameter checking issue: return value is -14, check the input parameters
* C++ call exception issue: C interface just returns the status value, without a detailed message. If you want details, print it in exception macro.

> **NOTE**: The exception from C interface is not the same as C++ exception, do not use the C status value in C++ debugging.

 ## See also
 * [OpenVINO™ README](../../../../README.md)
 * [C API developer guide](../README.md)
 * [OpenVINO Debug Capabilities](../../../../docs/dev/debug_capabilities.md)
