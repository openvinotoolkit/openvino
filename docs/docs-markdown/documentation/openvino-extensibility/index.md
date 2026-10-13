---
sidebar_position: 4
sidebar_label: "OpenVINO Extensibility Mechanism"
description: "Explore OpenVINO™ Extensibility API, which allows adding support for models with custom operations and their further implementation in applications."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenVINO Extensibility Mechanism



The Intel® Distribution of OpenVINO™ toolkit supports neural-network models trained with various frameworks, including
TensorFlow, PyTorch, ONNX, TensorFlow Lite, and PaddlePaddle. The list of supported operations is different for each of the supported frameworks.
To see the operations supported by your framework, refer to [Supported Framework Operations](../compatibility-and-support/supported-operations.md).

Custom operations, which are not included in the list, are not recognized by OpenVINO out-of-the-box. The need for custom operation may appear in two cases:

1. A new or rarely used regular framework operation is not supported in OpenVINO yet.
2. A new user operation that was created for some specific model topology by the author of the model using framework extension capabilities.

Importing models with such operations requires additional steps. This guide illustrates the workflow for running inference on models featuring custom operations. This allows plugging in your own implementation for them. OpenVINO Extensibility API enables adding support for those custom operations and using one implementation for model conversion API and OpenVINO Runtime.

Defining a new custom operation basically consists of two parts:

1. Definition of operation semantics in OpenVINO, the code that describes how this operation should be inferred consuming input tensor(s) and producing output tensor(s). The implementation of execution kernels for [GPU](custom-gpu-operations.md) is described in separate guides.
2. Mapping rule that facilitates conversion of framework operation representation to OpenVINO defined operation semantics.

The first part is required for inference. The second part is required for successful import of a model containing such operations from the original framework model format. There are several options to implement each part. The following sections will describe them in detail.

## Definition of Operation Semantics

If the custom operation can be mathematically represented as a combination of existing OpenVINO operations and such decomposition gives desired performance, then low-level operation implementation is not required. Refer to the latest OpenVINO operation set, when deciding feasibility of such decomposition. You can use any valid combination of existing operations. The next section of this document describes the way to map a custom operation.

If such decomposition is not possible or appears too bulky with a large number of constituent operations that do not perform well, then a new class for the custom operation should be implemented, as described in the [Custom Operation Guide](custom-openvino-operations.md).

You might prefer implementing a custom operation class if you already have a generic C++ implementation of operation kernel. Otherwise, try to decompose the operation first, as described above. Then, after verifying correctness of inference and resulting performance, you may move on to optional implementation of Bare Metal C++.

Additionally, it is also possible to implement custom operations using Python. OpenVINO provides a Python API that allows you to define and register custom operations. This can be particularly useful for rapid prototyping and testing of new operations.

## Mapping from Framework Operation

Mapping of custom operation is implemented differently, depending on model format used for import.
If a model is represented in the ONNX (including models exported from PyTorch in ONNX), TensorFlow Lite, PaddlePaddle or
TensorFlow formats, then you should use one of the classes from [Frontend Extension API](frontend-extensions.md),
the application of which is described below.

## Registering Extensions

A custom operation class and a new mapping frontend extension class object should be registered to be usable in OpenVINO runtime.

:::note
This documentation is derived from the [Template extension](https://github.com/openvinotoolkit/openvino/tree/master/src/core/template_extension), which demonstrates the details of extension development. It is based on minimalistic `Identity` operation that is a placeholder for your real custom operation. Review the complete, fully compilable code to see how it works.
:::

Use the `ov::Core::add_extension` method to load the extensions to the `ov::Core` object. This method allows loading library with extensions or extensions from the code.

### Load Extensions to Core

Extensions can be loaded from a code with the  `ov::Core::add_extension` method:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  core = ov.Core()

  # Use operation type to add operation extension
  core.add_extension(Identity)

  # or you can add operation extension object which is equivalent form
  core.add_extension(ov.OpExtension(Identity))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;

  // Use operation type to add operation extension
  core.add_extension<TemplateExtension::Identity>();

  // or you can add operation extension object which is equivalent form
  core.add_extension(ov::OpExtension<TemplateExtension::Identity>());
  ```
  </TabItem>
</Tabs>

The `Identity` is a custom operation class defined in [Custom Operation Guide](custom-openvino-operations.md). This is sufficient to enable reading OpenVINO IR which uses the `Identity` extension operation. In order to load original model directly to the runtime, add a mapping extension:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Register more sophisticated mapping with decomposition
  def conversion(node):
      input_node = node.get_input(0)
      return Identity(input_node).outputs()

  core.add_extension(ConversionExtension("Identity", conversion))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Register mapping for new frontends: FW's "TemplateIdentity" operation to TemplateExtension::Identity
  core.add_extension(ov::frontend::OpExtension<TemplateExtension::Identity>("Identity"));

  // Register more sophisticated mapping with decomposition
  core.add_extension(ov::frontend::ConversionExtension(
      "Identity",
      [](const ov::frontend::NodeContext& context) {
          // Arbitrary decomposition code here
          // Return a vector of operation outputs
          return ov::OutputVector{ std::make_shared<TemplateExtension::Identity>(context.get_input(0)) };
      }));
  ```
  </TabItem>
</Tabs>

If custom OpenVINO operation is implemented in C++ and loaded into the runtime through a shared library, there is no way to add a frontend mapping extension that refers to this custom operation. In this case, use C++ shared library approach to implement both operations semantics and framework mapping.


### Create a Library with Extensions {#create-a-library-with-extensions}

An extension library should be created in the following cases:

- Conversion of a model with custom operations in model conversion API
- Loading a model with custom operations in a Python application. This applies to both framework model and OpenVINO IR.
- Loading models with custom operations in tools that support loading extensions from a library, for example the `benchmark_app`.

To create an extension library, perform the following:

1. Create an entry point for extension library. OpenVINO provides the `OPENVINO_CREATE_EXTENSIONS()` macro, which allows to define an entry point to a library with OpenVINO Extensions.
This macro should have a vector of all OpenVINO Extensions as an argument.

Based on that, the declaration of an extension class might look like the following:

```cpp
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({

        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::Identity>>(),

        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::Identity>>()
    }));
```

2. Configure the build of your extension library, using the following CMake script:

```cpp
set(CMAKE_CXX_STANDARD 17)

set(TARGET_NAME "openvino_template_extension")

# The OpenVINO installed from PyPI can be used to find OpenVINO_DIR
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
            OUTPUT_VARIABLE OpenVINO_DIR_PY
            ERROR_QUIET)
    endif()
endif()

find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")

add_library(${TARGET_NAME} MODULE)

target_sources(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/identity.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/identity.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ov_extension.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

ov_build_target_faster(${TARGET_NAME} PCH)
```

This CMake script finds OpenVINO, using the `find_package` CMake command.

3. Build the extension library, running the commands below:

```bash
$ cd src/core/template_extension/new
$ mkdir build
$ cd build
$ cmake -DOpenVINO_DIR=<OpenVINO_DIR> ../
$ cmake --build .
```

The OpenVINO python distribution could be also used. The following code snippet demonstrates how to get the OpenVINO\_DIR:

```bash
$ cd src/core/template_extension/new
$ mkdir build
$ cd build
$ cmake -DOpenVINO_DIR=$(python3 -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')") ../
$ cmake --build .
```

4. After the build, you may use the path to your extension library to load your extensions to OpenVINO Runtime:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  core = ov.Core()
  # Load extensions library to ov.Core
  core.add_extension(path_to_extension_lib)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;
  // Load extensions library to ov::Core
  core.add_extension("openvino_template_extension.so");
  ```
  </TabItem>
</Tabs>

## See Also

- [OpenVINO Transformations](transformation-api/index.md)
- [Using OpenVINO Runtime Samples](../../get-started/learn-openvino/openvino-samples/index.md)
- [Hello Shape Infer SSD sample](../../get-started/learn-openvino/openvino-samples/hello-reshape-ssd.md)
