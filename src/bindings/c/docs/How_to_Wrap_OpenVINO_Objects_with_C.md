# How to wrap OpenVINO objects with C

Here is the details about how to wrap objects(C++) form OpenVINO to objects(C).

In OpenVINO C++ implementation, many objects are defined with `class`, such as `ov::Core`, `ov::Model`, `ov::InferRequest`..., but for C `class` doesn't be supported. So, C need to create new object to represent those objects. Three kinds of methods had been adopted in our implementation:
 * C `struct` contains a shared pointer to C++ `class`, [Wrap by C++ Shared Pointer](#wrap_by_c++_shared_pointer)
 * C `struct` contains a instance of C++ `class`, [Wrap by C++ Object](#wrap_by_c++_object)
 * C `struct` rewrite the C++ `class`, [Wrap by Rewrite](#wrap_by_rewrite)

In OpenVINO C++, most objects implemented with C++ `class` as following:
```ruby
class ClassName {
public:
    ...
private:
    ...
}
```

 ## Wrap by C++ Shared Pointer
C construct a new `struct` represents the class, which contains a shared pointer to the `class` as following:
```ruby
struct ov_ClassName {
    std::shared_ptr<ov::ClassName> object;
};
```
Here is an example (core) for wrapping by shared pointer:
`ov::Core` C++ [implementation](../../../inference/include/openvino/runtime/core.hpp)
```ruby
class OPENVINO_RUNTIME_API Core {
    class Impl;
    std::shared_ptr<Impl> _impl;
public:
    ...
}
```
Represent by C `struct` [here](../src/common.h)
```ruby
struct ov_core {
    std::shared_ptr<ov::Core> object;
};
```
C provides the `struct` by `typedef struct ov_core ov_core_t;`

 ## Wrap by C++ Object
C construct a new `struct` represents the class, which contains an instance to C++ `class` as following:
```ruby
struct ov_ClassName {
    ov::ClassName object;
};
```
Here is an example (layout) for wrapping by shared pointer:
`ov::Layout` C++ [implementation](../../../core/include/openvino/core/layout.hpp)
```ruby
class OPENVINO_API Layout {
public:
    /// \brief Constructs a dynamic Layout with no layout information.
    Layout();
    ...
}
```
Represent by C `struct` [here](../src/common.h)
```ruby
struct ov_layout {
    ov::Layout object;
};
```
C provides the `struct` by `typedef struct ov_layout ov_layout_t;`

 ## Wrap by Rewrite
C construct a new `struct` represents the class, which rewrites related info to the `class` as following:
```ruby
typedef struct {
    ov::ClassName object;
} ov_ClassName_t;
```
Here is an example (shape) for wrapping by shared pointer:
`ov::Shape` C++ [implementation](../../../core/include/openvino/core/shape.hpp)
```ruby
class Shape : public std::vector<size_t> {
public:
    OPENVINO_API Shape();
    ...
}
```
Represent by C `struct` [here](../src/common.h)
```ruby
typedef struct {
    int64_t rank;
    int64_t* dims;
} ov_shape_t;
```
> **NOTE**: this implementation needs developer create the C++ `class` based on the C `struct` info in using this rewrite type.

 ## See also
 * [Mapping Relationship of Objects](./docs/Mapping_Relationship_of_Objects.md)
 * [C API Reference](https://docs.openvino.ai/)