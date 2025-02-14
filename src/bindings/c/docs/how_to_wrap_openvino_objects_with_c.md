# How to wrap OpenVINO objects with C

Here is the details about how to wrap objects(C++) form OpenVINO to objects(C).

In OpenVINO C++ implementation, many objects are defined with `class`, such as `ov::Core`, `ov::Model`, `ov::InferRequest`..., but for C `class` doesn't be supported. So, C need to create new object to represent those objects. Three kinds of methods had been adopted in our implementation:
 * C `struct` contains a shared pointer to C++ `class`, [Wrap by C++ Shared Pointer](#wrap_by_c++_shared_pointer)
 * C `struct` contains a instance of C++ `class`, [Wrap by C++ Object](#wrap_by_c++_object)
 * C `struct` rewrite the C++ `class`, [Wrap by Rewrite](#wrap_by_rewrite)

Tips:
1) For the objects which needs to be hided for users, C `struct` contains a shared pointer will be adopted.
2) For the objects which needs to be created, operated and read by users, rewrite the C++ `class` will be better.
3) For some simple objects, C `struct` contains a instance of C++ `class` will be enough.

 ## Wrap by C++ Shared Pointer

C construct a new `struct` represents the class, which contains a shared pointer to the `class` as following:

```
struct ov_class_name {
    std::shared_ptr<ov::ClassName> object;
};
```

Here is an example (core) for wrapping by shared pointer:

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/inference/include/openvino/runtime/core.hpp#L41-L684

Represent by C `struct`:

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/bindings/c/src/common.h#L47-L53

C provides the `struct` by `typedef struct ov_core ov_core_t;`

 ## Wrap by C++ Object

C construct a new `struct` represents the class, which contains an instance to C++ `class` as following:

```
struct ov_ClassName {
    ov::ClassName object;
};
```

Here is an example (layout) for wrapping by shared pointer:

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/core/include/openvino/core/layout.hpp#L44-L107

Represent by C `struct`:

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/bindings/c/src/common.h#L95-L101

C provides the `struct` by `typedef struct ov_layout ov_layout_t;`

 ## Wrap by Rewrite

C construct a new `struct` represents the class, which rewrites related info to the `class` as following:

```
typedef struct {
    ov::ClassName object;
} ov_class_name_t;
```
Here is an example (shape) for wrapping by shared pointer:
https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/core/include/openvino/core/shape.hpp#L21-L40

Represent by C `struct` [here](../src/common.h)

https://github.com/openvinotoolkit/openvino/blob/d96c25844d6cfd5ad131539c8a0928266127b05a/src/bindings/c/include/openvino/c/ov_shape.h#L15-L22

> **NOTE**: this implementation needs developer create the C++ `class` based on the C `struct` info in using this rewrite type.

 ## See also
 * [OpenVINO™ README](../../../../README.md)
 * [C API developer guide](../README.md)
 * [C API Reference](https://docs.openvino.ai/2025/api/api_reference.html)