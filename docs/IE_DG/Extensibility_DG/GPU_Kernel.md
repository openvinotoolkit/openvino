# How to Implement Custom GPU Layers {#openvino_docs_IE_DG_Extensibility_DG_GPU_Kernel}

The GPU codepath abstracts many details about OpenCL&trade;. You need to provide the kernel code in OpenCL C and the configuration file that connects the kernel and its parameters to the parameters of the layer.

There are two options of using custom layer configuration file:

* Include a section with your kernels into the global automatically-loaded `cldnn_global_custom_kernels/cldnn_global_custom_kernels.xml` file, which is hosted in the `<INSTALL_DIR>/deployment_tools/inference_engine/bin/intel64/{Debug/Release}` folder
* Call the `InferenceEngine::Core::SetConfig()` method from your application with the `InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE` key and the configuration file name as a value before loading the network that uses custom layers to the plugin:

@snippet snippets/GPU_Kernel.cpp part0

All Inference Engine samples, except trivial `hello_classification`,
feature a dedicated command-line option `-c` to load custom kernels. For example, to load custom layers for the classification sample, run the command below:
```sh
$ ./classification_sample -m <path_to_model>/bvlc_alexnet_fp16.xml -i ./validation_set/daily/227x227/apron.bmp -d GPU
 -c <absolute_path_to_config>/custom_layer_example.xml
```

## Configuration File Format <a name="config-file-format"></a>

The configuration file is expected to follow the `.xml` file structure
with a node of the type `CustomLayer` for every custom layer you provide.

The definitions described in the sections below use the following notations:

Notation | Description
---|---
(0/1) | Can have 0 or 1 instances of this node/attribute
(1) | Must have only 1 instance of this node/attribute
(0+) | Can have any number of instances of this node/attribute
(1+) | Can have 1 or more instances of this node/attribute

### CustomLayer Node and Sub-node Structure

`CustomLayer` node contains the entire configuration for a single custom
layer.

| Attribute Name   |\#    |  Description |
|-----|-----|-----|
| `name`           | (1)  | The name of the layer type to be used. This name should be identical to the type used in the IR.|
| `type`           | (1)  | Must be `SimpleGPU`.                                                                             |
| `version`        | (1)  | Must be `1`.                                                                                   |

**Sub-nodes**: `Kernel` (1), `Buffers` (1), `CompilerOptions` (0+),
`WorkSizes` (0/1)

### Kernel Node and Sub-node Structure

`Kernel` node contains all kernel source code configuration. No kernel
node structure exists.

**Sub-nodes**: `Source` (1+), `Define` (0+)

### Source Node and Sub-node Structure

`Source` node points to a single OpenCL source file.

| Attribute Name | \#  ||
|-----|-----|-----|
| `filename`     | (1) | Name of the file containing OpenCL source code. Notice that path is relative to your executable. Multiple source nodes will have their sources concatenated in order. |

**Sub-nodes**: None

### Define Node and Sub-node Structure

`Define` node configures a single `#&zwj;define` instruction to be added to
the sources during compilation (JIT).

| Attribute Name | \#    | Description |
|------|-------|------|
| `name`         | (1)   | The name of the defined JIT. For static constants, this can include the value as well (taken as a string). |
| `param`        | (0/1) | This parameter value is used as the value of this JIT definition.                                     |
| `type`         | (0/1) | The parameter type. Accepted values: `int`, `float`, and `int[]`, `float[]` for arrays.                    |
| `default`      | (0/1) | The default value to be used if the specified parameters is missing from the layer in the IR.              |

**Sub-nodes:** None

The resulting JIT has the following form:
`#&zwj;define [name] [type] [value/default]`.

### Buffers Node and Sub-node Structure

`Buffers` node configures all input/output buffers for the OpenCL entry
function. No buffers node structure exists.

**Sub-nodes:** `Data` (0+), `Tensor` (1+)

### Data Node and Sub-node Structure

`Data` node configures a single input with static data (for example,
weights or biases).

| Attribute Name | \#  | Description |
|----|-----|------|
| `name`         | (1) | Name of a blob attached to a layer in the IR             |
| `arg-index`    | (1) | 0-based index in the entry function arguments to be bound to |

**Sub-nodes**: None

### Tensor Node and Sub-node Structure

`Tensor` node configures a single input or output tensor.

| Attribute Name | \#    | Description  |
|------|-------|-------|
| `arg-index`    | (1)   | 0-based index in the entry function arguments to be bound to.                                                                          |
| `type`         | (1)   | `input` or `output`                                                                                                                    |
| `port-index`   | (1)   | 0-based index in the layer’s input/output ports in the IR                                                                              |
| `format`       | (0/1) | Data layout declaration for the tensor. Accepted values: `BFYX`, `BYXF`, `YXFB`, `FYXB` (also in all lowercase). Default value: `BFYX` |

### CompilerOptions Node and Sub-node Structure

`CompilerOptions` node configures the compilation flags for the OpenCL
sources.

| Attribute Name | \#  | Description                                        |
|--------|-----|------|
| `options`      | (1) | Options string to be passed to the OpenCL compiler |

**Sub-nodes**: None

### WorkSizes Node and Sub-node Structure

`WorkSizes` node configures the global/local work sizes to be used when
queuing the OpenCL program for execution.

| Attribute Name      | \#             | Description                                                                 |
|-----|------|-----|
| `global`<br>`local` | (0/1)<br>(0/1) | An array of up to 3 integers (or formulas) for defining the OpenCL work-sizes to be used during execution.<br> The formulas can use the values of the B,F,Y,X dimensions and contain the operators: +,-,/,\*,% (all evaluated in integer arithmetic). <br>Default value: `global=”B*F*Y*X” local=””` |
| `dim`               | (0/1)          | A tensor to take the work size from. Accepted values: `input N`, `output`, where `N` is an index of input tensor starting with 0. Default value: `output` |

**Sub-nodes**: None

## Example Configuration File

The following code sample provides an example configuration file (in the
`.xml` format). For information on configuration file structure, see
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

## Built-In Defines for Custom Layers

The following table includes definitions that are attached before
the user sources, where `<TENSOR>` is the actual input and output, for
example, `INPUT0` or `OUTPUT0`.

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
| `<TENSOR>_FORMAT_` | The format of the tensor, BFYX, BYXF, YXFB , FYXB, or ANY. The format is concatenated to the defined name. You can use the tensor format to define codepaths in your code with `#&zwj;ifdef/#&zwj;endif`. |
| `<TENSOR>_LOWER_PADDING` | An array of padding elements used for the tensor dimensions before they start. Always ordered as BFYX.|
| `<TENSOR>_ LOWER_PADDING_SIZE` | The size of the `<TENSOR>_LOWER_PADDING` array  |
| `<TENSOR>_UPPER_PADDING`   | An array of padding elements used for the tensor dimensions after they end. Always ordered as BFYX. |
| `<TENSOR>_UPPER_PADDING_SIZE`  | The size of the `<TENSOR>_UPPER_PADDING` array |
| `<TENSOR>_PITCHES` | The number of elements between adjacent elements in each dimension. Always ordered as BFYX.|
| `<TENSOR>_PITCHES_SIZE`| The size of the `<TENSOR>_PITCHES` array   |
| `<TENSOR>_OFFSET`| The number of elements from the start of the tensor to the first valid element (bypassing the lower padding)   |
All `<TENSOR>` values are automatically defined for every tensor
bound to this layer (`INPUT0`, `INPUT1`, `OUTPUT0`, and so on), as shown
in the following for example:

```sh
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
    const uint idbf = get_global_id(2);//batches*features, as OpenCL supports 3D nd-ranges only
    const uint feature = idbf%OUTPUT0_DIMS[1];
    const uint batch   = idbf/OUTPUT0_DIMS[1];
    //notice that pitches are in elements, not in bytes!
    const uint in_id  = batch*INPUT0_PITCHES[0] + feature*INPUT0_PITCHES[1]   + idy*INPUT0_PITCHES[2]  + idx*INPUT0_PITCHES[3]  + INPUT0_OFFSET;
    const uint out_id = batch*OUTPUT0_PITCHES[0] + feature*OUTPUT0_PITCHES[1]  + idy*OUTPUT0_PITCHES[2]  + idx*OUTPUT0_PITCHES[3]  + OUTPUT0_OFFSET;

    INPUT0_TYPE value = input0[in_id];
    //neg_slope (which is non-zero for leaky ReLU) is put automatically as #define, refer to the config xml
    output[out_id] = value < 0 ? value * neg_slope : value;
}
```

> **NOTE:** As described in the previous section, all the things like
> `INPUT0_TYPE` are actually defined as OpenCL (pre-)compiler inputs by
> the Inference Engine for efficiency reasons. See [Debugging
> Tips](#debugging-tips) for information on debugging the results.

> **NOTE**: Several GPU-targeted kernels are also added to the binaries upon samples compilation
> so that the sample application can easy load them.
> Refer to the `cldnn_global_custom_kernels` folder in the GPU plugin installation directory.

## Debugging Tips<a name="debugging-tips"></a>

* **Dumping the Resulting Kernels**.
It is recommended to get a dump of the kernel with all of
the values set by the Inference Engine, such as tensor sizes,
floating-point, and integer kernel parameters. To get the dump, add the
following line to your code that configures the GPU plugin to output the
custom kernels:

@snippet snippets/GPU_Kernel.cpp part1

When the Inference Engine compiles the kernels for the specific network,
it also outputs the resulting code for the custom kernels. In the
directory of your executable, find files like
`clDNN_program0.cl`, `clDNN_program1.cl`. There are as many files as
distinct sets of parameters for your custom kernel: different input
tensor sizes and kernel parameters.

* **Using `printf` in the OpenCL™ Kernels**.
To debug the specific values, you can use `printf` in your kernels.
However, be careful: for instance, do not output excessively
as it would generate too much data. The `printf` output is typical, so
your output can be truncated to fit the buffer. Also, because of
buffering, you actually get an entire buffer of output when the
execution ends.<br>
For more information, refer to the [printf
Function](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/printfFunction.html).
