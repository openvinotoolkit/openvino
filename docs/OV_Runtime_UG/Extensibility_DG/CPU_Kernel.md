# CPU Kernel Custom Operations {#openvino_docs_IE_DG_Extensibility_DG_CPU_Kernel}

To enable operations not supported by OpenVINO™ out of the box, you need a custom extension for Model Optimizer, a custom nGraph operation set, and a custom kernel for the device you will target. This page describes custom kernel support for the CPU device.

The primary means of the performance of the CPU codepath in the Inference Engine is the Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN), and new CPU kernels extend the Inference Engine plugin for the Intel MKL-DNN. Implementing the InferenceEngine::ILayerExecImpl API call defines a general CPU-side extension. There are no Intel MKL-DNN specifics in the way you need to implement a kernel.

## Implementation Class

All custom kernels for the CPU plugin should be inherited from the InferenceEngine::ILayerExecImpl interface.
Based on that, declaration of a kernel implementation class can look as follows:

@snippet template_extension/old/cpu_kernel.hpp cpu_implementation:header

### Class Fields

The provided implementation has several fields:

 * `add` of the type `int64_t` is an attribute of a custom operation.
 * `inShape` of the type `ngraph::Shape` is an input shape.
 * `outShape` of the type `ngraph::Shape` is an output shape.
 * `error` of the type `std::string` is a field to handle errors from a constructor.

### Constructor of Implementation

An implementation constructor checks parameters of an nGraph operation, stores required attributes, and stores an error message in case of an error.

@snippet template_extension/old/cpu_kernel.cpp cpu_implementation:ctor

### `getSupportedConfigurations`

The InferenceEngine::ILayerExecImpl::getSupportedConfigurations method returns all supported configuration formats (input/output tensor layouts) for your implementation. To specify formats of data, use InferenceEngine::TensorDesc. Refer to the [Memory Primitives](../Memory_primitives.md) section for instructions.

@snippet template_extension/old/cpu_kernel.cpp cpu_implementation:getSupportedConfigurations

### `init`

The InferenceEngine::ILayerExecImpl::init method gets a runtime-selected configuration from a vector that is populated from the `getSupportedConfigurations` method and checks the parameters:

@snippet template_extension/old/cpu_kernel.cpp cpu_implementation:init

### `execute`

The InferenceEngine::ILayerExecImpl::execute method accepts and processes the actual tensors as input/output blobs:

@snippet template_extension/old/cpu_kernel.cpp cpu_implementation:execute

## Register Implementation in `Extension` Class

To register custom kernel implementation in the [Extension](Extension.md) class, implement the following methods:

* <a href="#getImpTypes">getImplTypes</a>
* <a href="#getImplementation">getImplementation</a>

### <a name="getImpTypes"><code>getImplTypes</code></a>

InferenceEngine::IExtension::getImplTypes returns a vector of implementation types for an operation.

@snippet template_extension/old/extension.cpp extension:getImplTypes

### <a name="getImplementation"><code>getImplementation</code></a>

InferenceEngine::IExtension::getImplementation returns the kernel implementation with a specified type for an operation.

@snippet template_extension/old/extension.cpp extension:getImplementation


## Load Extension with Executable Kernels to Plugin

Use the `AddExtension` method of the general plugin interface to load your primitives:

@snippet snippets/CPU_Kernel.cpp part0
