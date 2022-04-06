# How to Implement Custom GPU Operations {#openvino_docs_Extensibility_UG_GPU}

To enable operations not supported by OpenVINO out of the box, you may need an extension for OpenVINO operation set, and a custom kernel for the device you will target. This page describes custom kernel support for the GPU device.

The GPU codepath abstracts many details about OpenCL\*. You need to provide the kernel code in OpenCL C and an XML configuration file that connects the kernel and its parameters to the parameters of the operation.

There are two options for using the custom operation configuration file:

* Include a section with your kernels into the automatically-loaded `<lib_path>/cldnn_global_custom_kernels/cldnn_global_custom_kernels.xml` file.
* Call the `ov::Core::set_property()` method from your application with the `"CONFIG_FILE"` key and the configuration file name as a value before loading the network that uses custom operations to the plugin:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/custom_kernels_api.cpp part0
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/custom_kernels_api.py part0
@endsphinxtab

@endsphinxtabset

All OpenVINO samples, except the trivial `hello_classification`, and most Open Model Zoo demos
feature a dedicated command-line option `-c` to load custom kernels. For example, to load custom operations for the classification sample, run the command below:
```sh
$ ./classification_sample -m <path_to_model>/bvlc_alexnet_fp16.xml -i ./validation_set/daily/227x227/apron.bmp -d GPU
 -c <absolute_path_to_config>/custom_layer_example.xml
```

## Configuration File Format <a name="config-file-format"></a>

The configuration file is expected to follow the `.xml` file structure
with a node of the type `CustomLayer` for every custom operation you provide.

The definitions described in the sections below use the following notations:

Notation | Description
---|---
(0/1) | Can have zero or one instance of this node or attribute
(1) | Must have only one instance of this node or attribute
(0+) | Can have any number of instances of this node or attribute
(1+) | Can have one or more instances of this node or attribute

### CustomLayer Node and Sub-Node Structure

`CustomLayer` node contains the entire configuration for a single custom operation.

| Attribute Name   |\#    |  Description |
|-----|-----|-----|
| `name`           | (1)  | The name of the operation type to be used. This name should be identical to the type used in the IR.|
| `type`           | (1)  | Must be `SimpleGPU`.                                                                                |
| `version`        | (1)  | Must be `1`.                                                                                        |

**Sub-nodes**: `Kernel` (1), `Buffers` (1), `CompilerOptions` (0+),
`WorkSizes` (0/1)

### Kernel Node and Sub-Node Structure

`Kernel` node contains all kernel source code configuration.

**Sub-nodes**: `Source` (1+), `Define` (0+)

### Source Node and Sub-Node Structure

`Source` node points to a single OpenCL source file.

| Attribute Name | \#  |Description|
|-----|-----|-----|
| `filename`     | (1) | Name of the file containing OpenCL source code. Note that the path is relative to your executable. Multiple source nodes will have their sources concatenated in order. |

**Sub-nodes**: None

### Define Node and Sub-Node Structure

`Define` node configures a single `#&zwj;define` instruction to be added to
the sources during compilation (JIT).

| Attribute Name | \#    | Description |
|------|-------|------|
| `name`         | (1)   | The name of the defined JIT. For static constants, this can include the value as well, which is taken as a string. |
| `param`        | (0/1) | This parameter value is used as the value of this JIT definition.                                          |
| `type`         | (0/1) | The parameter type. Accepted values: `int`, `float`, and `int[]`, `float[]` for arrays.                    |
| `default`      | (0/1) | The default value to be used if the specified parameters are missing from the operation in the IR.          |

**Sub-nodes:** None

The resulting JIT has the following form:
`#&zwj;define [name] [type] [value/default]`.

### Buffers Node and Sub-Node Structure

`Buffers` node configures all input/output buffers for the OpenCL entry
function. No buffers node structure exists.

**Sub-nodes:** `Data` (0+), `Tensor` (1+)

### Data Node and Sub-Node Structure

`Data` node configures a single input with static data, for example,
weights or biases.

| Attribute Name | \#  | Description |
|----|-----|------|
| `name`         | (1) | Name of a blob attached to an operation in the IR             |
| `arg-index`    | (1) | 0-based index in the entry function arguments to be bound to |

**Sub-nodes**: None

### Tensor Node and Sub-Node Structure

`Tensor` node configures a single input or output tensor.

| Attribute Name | \#    | Description  |
|------|-------|-------|
| `arg-index`    | (1)   | 0-based index in the entry function arguments to be bound to.                                                                          |
| `type`         | (1)   | `input` or `output`                                                                                                                    |
| `port-index`   | (1)   | 0-based index in the operation input/output ports in the IR                                                                            |
| `format`       | (0/1) | Data layout declaration for the tensor. Accepted values: `BFYX`, `BYXF`, `YXFB`, `FYXB`, and same values in all lowercase. Default value: `BFYX` |

### CompilerOptions Node and Sub-Node Structure

`CompilerOptions` node configures the compilation flags for the OpenCL
sources.

| Attribute Name | \#  | Description                                        |
|--------|-----|------|
| `options`      | (1) | Options string to be passed to the OpenCL compiler |

**Sub-nodes**: None

### WorkSizes Node and Sub-Node Structure

`WorkSizes` node configures the global/local work sizes to be used when
queuing an OpenCL program for execution.

| Attribute Name      | \#             | Description                                                                 |
|-----|------|-----|
| `global`<br>`local` | (0/1)<br>(0/1) | An array of up to three integers or formulas for defining OpenCL work-sizes to be used during execution.<br> The formulas can use the values of the B,F,Y,X dimensions and contain the operators: +,-,/,\*,%. All operators are evaluated in integer arithmetic. <br>Default value: `global=”B*F*Y*X” local=””` |
| `dim`               | (0/1)          | A tensor to take the work-size from. Accepted values: `input N`, `output`, where `N` is an index of input tensor starting with 0. Default value: `output` |

**Sub-nodes**: None

## Example Configuration File

The following code sample provides an example configuration file in XML
format. For information on the configuration file structure, see
[Configuration File Format](#config-file-format).
```xml
<CustomLayer name="ReLU" type="SimpleGPU" version="1">
  <Kernel entry="example_relu_kernel">
    <Source filename="custom_layer_kernel.cl"/>
    <Define name="neg_slope" type="float" param="negative_slope" default="0.0"/>
  </Kernel>
  <Buffers>
    <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
    <Tensor arg-index="1" type="output" port-index="0" format="BFYX"/>
  </Buffers>
  <CompilerOptions options="-cl-mad-enable"/>
  <WorkSizes global="X,Y,B*F"/>
</CustomLayer>
```

## Built-In Definitions for Custom Layers

The following table includes definitions that are attached before
user sources.

For an example, see [Example Kernel](#example-kernel).

| Name | Value  |
|---|---|
| `NUM_INPUTS` | Number of the input tensors bound to this kernel |
| `GLOBAL_WORKSIZE`  | An array of global work sizes used to execute this kernel |
| `GLOBAL_WORKSIZE_SIZE` | The size of the `GLOBAL_WORKSIZE` array |
| `LOCAL_WORKSIZE`  | An array of local work sizes used to execute this kernel  |
| `LOCAL_WORKSIZE_SIZE`   | The size of the `LOCAL_WORKSIZE` array |
| `<TENSOR>_DIMS`| An array of the tensor dimension sizes. Always ordered as `BFYX` |
| `<TENSOR>_DIMS_SIZE`| The size of the `<TENSOR>_DIMS` array.|
| `<TENSOR>_TYPE`| The datatype of the tensor: `float`, `half`, or `char`|
| `<TENSOR>_FORMAT_<TENSOR_FORMAT>` | The format of the tensor, BFYX, BYXF, YXFB , FYXB, or ANY. The format is concatenated to the defined name. You can use the tensor format to define codepaths in your code with `#&zwj;ifdef/#&zwj;endif`. |
| `<TENSOR>_LOWER_PADDING` | An array of padding elements used for the tensor dimensions before they start. Always ordered as BFYX.|
| `<TENSOR>_LOWER_PADDING_SIZE` | The size of the `<TENSOR>_LOWER_PADDING` array  |
| `<TENSOR>_UPPER_PADDING`   | An array of padding elements used for the tensor dimensions after they end. Always ordered as BFYX. |
| `<TENSOR>_UPPER_PADDING_SIZE`  | The size of the `<TENSOR>_UPPER_PADDING` array |
| `<TENSOR>_PITCHES` | The offset (in elements) between adjacent elements in each dimension. Always ordered as BFYX.|
| `<TENSOR>_PITCHES_SIZE`| The size of the `<TENSOR>_PITCHES` array   |
| `<TENSOR>_OFFSET`| The number of elements from the start of the tensor to the first valid element, bypassing the lower padding.  |

All `<TENSOR>` values are automatically defined for every tensor
bound to this operation, such as `INPUT0`, `INPUT1`, and `OUTPUT0`, as shown
in the following example:

```c
#define INPUT0_DIMS_SIZE 4
#define INPUT0_DIMS (int []){ 1,96,55,55, }
```

## Example Kernel<a name="example-kernel"></a>

```c
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void example_relu_kernel(
    const __global INPUT0_TYPE*  input0,
          __global OUTPUT0_TYPE* output)
{
    const uint idx  = get_global_id(0);
    const uint idy  = get_global_id(1);
    const uint idbf = get_global_id(2); // batches*features, as OpenCL supports 3D nd-ranges only
    const uint feature = idbf % OUTPUT0_DIMS[1];
    const uint batch   = idbf / OUTPUT0_DIMS[1];
    //notice that pitches are in elements, not in bytes!
    const uint in_id  = batch*INPUT0_PITCHES[0] + feature*INPUT0_PITCHES[1]   + idy*INPUT0_PITCHES[2]  + idx*INPUT0_PITCHES[3]  + INPUT0_OFFSET;
    const uint out_id = batch*OUTPUT0_PITCHES[0] + feature*OUTPUT0_PITCHES[1]  + idy*OUTPUT0_PITCHES[2]  + idx*OUTPUT0_PITCHES[3]  + OUTPUT0_OFFSET;

    INPUT0_TYPE value = input0[in_id];
    // neg_slope (which is non-zero for leaky ReLU) is put automatically as #define, refer to the config xml
    output[out_id] = value < 0 ? value * neg_slope : value;
}
```


> **NOTE**: As described in the previous section, all items like
> `INPUT0_TYPE` are actually defined as OpenCL (pre-)compiler inputs by
> the OpenVINO for efficiency reasons. See [Debugging
> Tips](#debugging-tips) for information on debugging the results.

## Debugging Tips<a name="debugging-tips"></a>

* **Using `printf` in the OpenCL™ Kernels**.
To debug the specific values, you can use `printf` in your kernels.
However, be careful not to output excessively, which
could generate too much data. The `printf` output is typical, so
your output can be truncated to fit the buffer. Also, because of
buffering, you actually get an entire buffer of output when the
execution ends.<br>

For more information, refer to the [printf
Function](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/printfFunction.html).
