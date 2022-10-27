# How to Write Unite Test for C API

To ensure the accuracy of C API, all interfaces need to implement function level unite test at least. According to the object define, unite test cases are classified into following components:

| Unite Case Component | Location | Description |
|:---     |:---   |:---
|Core|[ov_core_test.cpp](../tests/ov_core_test.cpp)| including all core related interfaces tests
|Model|[ov_model_test.cpp](../tests/ov_model_test.cpp)| including all model related interfaces tests
|Compiled Model|[ov_compiled_model_test.cpp](../tests/ov_compiled_model_test.cpp)| including all compiled model related interfaces tests
|Infer Request|[ov_infer_request_test.cpp](../tests/ov_infer_request_test.cpp)| including all infer request related interfaces tests
|Tensor|[ov_tensor_test.cpp](../tests/ov_tensor_test.cpp)| including all tensor related interfaces tests
|Partial Shape|[ov_partial_shape_test.cpp](../tests/ov_partial_shape_test.cpp)| including all partial shape related interfaces tests
|Layout|[ov_layout_test.cpp](../tests/ov_layout_test.cpp)| including all layout related interfaces tests
|Preprocess|[ov_preprocess_test.cpp](../tests/ov_preprocess_test.cpp)| including all preprocess related interfaces tests


If developer wrap new interfaces from OpenVINO C++, you also need add the unite test case in the correct location.
Here is an example for C interface unite test case:
* C++ interface for read model,

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/inference/include/openvino/runtime/core.hpp#L87-L100 

* C wrap this interface like,

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/bindings/c/src/ov_core.cpp#L71-L90

* Create unite test case for this interface. At first, this interface is for core operation so the location should at [ov_core_test.cpp](../tests/ov_core_test.cpp). Also, the interface has default parameter so need to make unite test case for parameter missing. The final based function level test like:

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/bindings/c/tests/ov_core_test.cpp#L39-L63




