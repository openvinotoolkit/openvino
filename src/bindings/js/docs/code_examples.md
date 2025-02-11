# How to extend the OpenVINO™ JavaScript API code

## Build the OpenVINO™ JavaScript API

For detailed build instructions, refer to the [OpenVINO™ JavaScript API documentation](./README.md).


## Project's naming conventions

When implementing the C++ sources for the JavaScript API, it is essential to adhere to the OpenVINO naming conventions described in the [OpenVINO Coding Style Guide](../../../../docs/dev/coding_style.md). In summary, the naming style employs `Snake Case` for methods, functions, and variables, while `Camel Case` is used for class names. Additionally, the naming of entities in the C++ sources should closely mirror their equivalents in the C++ API to maintain consistency.

For methods that are exposed to JavaScript, the naming convention transitions to `Camel Case`, aligning with common JavaScript practices. As an example, a method in the C++ API named `get_element_type` would be represented in the JavaScript API as `getElementType()`.


## node-addon-api module

[node addon api](https://github.com/nodejs/node-addon-api) is used to create OpenVINO JavaScript API for Node.js. The quickest way to learn is to follow the official [examples](https://github.com/nodejs/node-addon-examples). It is recommended to check out the tutorial on [how to create a JavaScript object from a C++ object](https://github.com/nodejs/node-addon-examples/tree/main/src/2-js-to-native-conversion/object-wrap-demo/node-addon-api).


## Adding a new class and method

To introduce a new `MyTensor` class that interacts with the `ov::Tensor` class, follow these steps:
 - The class should facilitate construction from an ov::Tensor instance and allow initialization from a JavaScript element type and shape.
 - It should also provide a getElementType method that retrieves the ov::Tensor element type.

Begin by creating a header file for the `MyTensor` class in the OpenVINO repository at `<openvino_repo>/src/bindings/js/node/include/my_tensor.hpp`. This file should contain the necessary includes and class definitions:
```cpp
class MyTensor : public Napi::ObjectWrap<MyTensor> {
public:
    // Constructor for the wrapper class
    MyTensor(const Napi::CallbackInfo& info);

    // It returns a JavaScript class definition
    static Napi::Function get_class(Napi::Env env);

    // It returns the element type of ov::Tensor
    Napi::Value get_element_type(const Napi::CallbackInfo& info);

private:
    ov::Tensor _tensor;
};
```
The implementation of the class methods should be placed in a source file at `<openvino_repo>/src/bindings/js/node/src/my_tensor.cpp`
```cpp
MyTensor::MyTensor(const Napi::CallbackInfo& info) : Napi::ObjectWrap<MyTensor>(info) {
    std::vector<std::string> allowed_signatures;

    try {
        if (ov::js::validate<Napi::String, Napi::Array>(info, allowed_signatures)) {
            const auto type = js_to_cpp<ov::element::Type_t>(info, 0);
            const auto& shape = js_to_cpp<ov::Shape>(info, 1);
            this->_tensor = ov::Tensor(type, shape);
        } else {
            OPENVINO_THROW("'MyTensor' constructor", ov::js::get_parameters_error_msg(info, allowed_signatures));
        }
    } catch (std::exception& err) {
        reportError(info.Env(), err.what());
    }
}

Napi::Function MyTensor::get_class(Napi::Env env) {
    return DefineClass(env,
                       "MyTensor",
                       {
                           InstanceMethod("getElementType", &MyTensor::get_element_type),
                       });
}

Napi::Value MyTensor::get_element_type(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), _tensor.get_element_type().to_string());
}
```
Finally, update the `CMakeLists.txt` file at `<openvino_repo>/src/bindings/js/node/` to include the new source file in the build process:
```cmake
add_library(${PROJECT_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/my_tensor.cpp
)
```


### Argument validation and conversion

When binding JavaScript arguments with C++ functions, it is crucial to validate and convert the arguments appropriately. The template `ov::js::validate` function is a utility that facilitates this process. It is particularly useful for handling different overloads of functions and ensuring standardized error messages when arguments do not match expected signatures.
Before implementing a new conversion function, such as `js_to_cpp<ov::Shape>`, review the existing [helper methods](../../node/include/helper.hpp) to see if one already meets your requirements.


### New class initialization

When a new class is introduced to the `openvino-node` module, it must be initialized upon module loading. This is done in the [addon.cpp](../../src/addon.cpp) file. The initialization process registers the class with the Node.js environment so that it can be used within JavaScript code.
```cpp
Napi::Object init_module(Napi::Env env, Napi::Object exports) {
    auto addon_data = new AddonData();
    env.SetInstanceData<AddonData>(addon_data);
    init_class(env, exports, "MyTensor", &MyTensor::get_class, addon_data->my_tensor);

    return exports;
}
```
To keep track of the JavaScript class definitions, they are kept in `<openvino_repo>/src/bindings/js/node/include/addon.hpp`.
```cpp
struct AddonData {
    Napi::FunctionReference my_tensor;
     // and other class references
};
```

### Document the new functionality

The last step is to add the TypeScript type definitions and describe the new functionality.
```typescript
/**
 * The {@link MyTensor} class and its documentation.
 */
interface MyTensor {
  /**
   * It gets the tensor element type.
   */
  getElementType(): element;

}
interface MyTensorConstructor {
  /**
   * It constructs a tensor using the element type and shape. The new tensor
   * data will be allocated by default.
   * @param type The element type of the new tensor.
   * @param shape The shape of the new tensor.
   */
  new(type: element | elementTypeString, shape: number[]): MyTensor;
}

export interface NodeAddon {
  MyTensor: MyTensorConstructor,
}
```


## Testing the new code

Now that coding is finished, remember to rebuild the project and test it out.

To learn how to test your code, refer to the guide on [how to test OpenVINO™ JavaScript API.](./test_examples.md)

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO™ bindings README](../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
