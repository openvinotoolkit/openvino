# How to Implement Custom Layers for VPU (Intel® Neural Compute Stick 2) {#openvino_docs_IE_DG_Extensibility_DG_VPU_Kernel}

To enable operations not supported by OpenVINO™ out of the box, you need a custom extension for Model Optimizer, a custom nGraph operation set, and a custom kernel for the device you will target. This page describes custom kernel support for one the VPU, the Intel® Neural Compute Stick 2 device, which uses the MYRIAD device plugin.

> **NOTES:** 
> * OpenCL\* custom layer support is available in the preview mode.
> * This section assumes you are familiar with developing kernels using OpenCL.
To customize your topology with an OpenCL layer, carry out the tasks described on this page:

1. Write and compile your OpenCL code with the standalone offline OpenCL compiler (`clc`).
2. Write a configuration file to bind the OpenCL kernel to the topology file (`.xml`) of the model IR.
3. Pass the configuration file to the OpenVINO™ Runtime with the model IR.

## Compile OpenCL code for VPU (Intel® Neural Compute Stick 2)

> **NOTE**: OpenCL compiler, targeting Intel® Neural Compute Stick 2 for the SHAVE* processor only, is redistributed with OpenVINO.
OpenCL support is provided by ComputeAorta* and is distributed under a license agreement between Intel® and Codeplay* Software Ltd.
The OpenCL toolchain for the Intel® Neural Compute Stick 2 supports offline compilation only, so first compile OpenCL C code using the standalone `clc` compiler. You can find the compiler binary at `<INSTALL_DIR>/tools/cl_compiler`.

> **NOTE**: By design, custom OpenCL layers support any OpenCL kernels written assuming OpenCL version 1.2. It also supports half float extension and is optimized for this type, because it is a native type for Intel® Movidius™ VPUs.
1. Prior to running a compilation, make sure that the following variables are set:
   * `SHAVE_MA2X8XLIBS_DIR=<INSTALL_DIR>/tools/cl_compiler/lib/`
   * `SHAVE_LDSCRIPT_DIR=<INSTALL_DIR>/tools/cl_compiler/ldscripts/`
   * `SHAVE_MYRIAD_LD_DIR=<INSTALL_DIR>/tools/cl_compiler/bin/`
   * `SHAVE_MOVIASM_DIR=<INSTALL_DIR>/tools/cl_compiler/bin/`
2. Run the compilation with the command below. You should use `--strip-binary-header` to make an OpenCL runtime-agnostic binary runnable with the OpenVINO™ Runtime.
   ```bash
   cd <INSTALL_DIR>/tools/cl_compiler/bin
   ./clc --strip-binary-header custom_layer.cl -o custom_layer.bin
   ```

## Write a Configuration File

To tie the topology IR for a layer you customize, prepare a configuration file, so that the OpenVINO™ Runtime can find parameters for your kernel and the execution work grid is described.
For example, consider the following OpenCL kernel signature:
```cpp
__kernel void reorg_nhwc(__global const half *src, __global half *out, int w, int h, int c, int stride);
```
A configuration file for this kernel might be the following:
```xml
<CustomLayer name="ReorgYolo" type="MVCL" version="1">
   <Kernel entry="reorg_nhwc">
       <Source filename="reorg.bin"/>
   </Kernel>
   <Parameters>
       <Tensor arg-name="src"    type="input"  port-index="0"                format="BYXF"/>
       <Tensor arg-name="out"    type="output" port-index="0"                format="BYXF"/>
       <Scalar arg-name="w"      type="int"    port-index="0" source="I.X"                />
       <Scalar arg-name="h"      type="int"    port-index="0" source="I.Y"                />
       <Scalar arg-name="c"      type="int"    port-index="0" source="I.F"                />
       <Scalar arg-name="stride" type="int"                   source="stride"             />
   </Parameters>
   <WorkSizes dim="input,0" global="(Y+7)/8*8,1,1" local="8,1,1"/>
</CustomLayer>
```
Each custom layer is described with the `CustomLayer` node. It has the following nodes and attributes:
  - Root node `CustomLayer` contains the following attributes:
    - `name` – (Required) The name of the OpenVINO™ Runtime layer to bind the kernel with.
    - `type` and `version` – (Required) Reserved for future use. Set them to `MVCL` and `1` respectively.
    - `max-shaves` – (Optional) The maximum number of SHAVE cores that should be dedicated for the layer. It is useful for debugging concurrency issues or for resource saving that memory bound kernel does not scale well with the number of cores, so more resources can be left for the rest of a topology.
  - Sub-node `Kernel` must contain the following attributes:
    - `entry` – The name of your kernel function as you defined it in a source file. In the example above, it is `reorg_nhwc`.
    - Node `Source` must contain the following attributes:
      - `filename` – The path to a compiled binary relative to the XML configuration file.
  - Sub-node `Parameters` – Describes parameters bindings. For more information, see the description below.
  - Sub-node `WorkSizes` – Describes local and global work group sizes and the source for dimension deduction as a pair `direction,port`. In the example above, the work group is described relatively to the dimension of the input tensor that comes through port 0 in the IR. `global` and `local` work group configurations support any simple math expressions with +,-,\*,/, and () from `B`(batch), `Y`(height), `X`(width) and `F`(channels).
  - Sub-node `Where` – Allows to customize bindings with the `key="value"` attribute. For example, to substitute only 3x3 convolutions, write `<Where kernel="3,3"/>` in the binding xml.

  Parameter description supports `Tensor` of one of tensor types such as `input`, `output`, `input_buffer`, `output_buffer` or `data`, `Scalar`, or `Data` nodes and has the following format:
  - Each `Tensor` node of `input` or `output` type must contain the following attributes:
    - `arg-name` – The name of a kernel parameter in the kernel signature.
    - `type` – Node type: `input` or `output` as specified in the IR.
    - `port-index` – A number of input/output ports as specified in the IR.
    - `format` – The channel order in the tensor. Optional conversion layers are generated if the custom layer format is not compatible with formats of neighboring layers. `BFXY`, `BYXF`, and `ANY` formats are supported currently.
  - Each `Tensor` node of `input_buffer` or `output_buffer` type must contain the following attributes:
    - `arg-name` – The name of a kernel parameter in the kernel signature.
    - `type` – Node type: `input_buffer` or `output_buffer`. Use the appropriate type to bind multiple kernels that correspond to different stages of the same layer.
    - `port-index` – The unique identifier to bind by.
    - `dim` – The dim source with the same `direction,port` format used for `WorkSizes` bindings.
    - `size` – Amount of bytes needed. Current expression syntax supports only expression over dimensions of over selected input/output tensor or constants and might be expended in the future.

    Here is an example of multi-stage MVN layer binding:
  ```xml
  <CustomLayer name="MVN" stage="0" type="MVCL" version="1">
      <Kernel entry="reduction_mean">
          <Source filename="mvn.bin"/>
      </Kernel>
      <Parameters>
          <Tensor arg-name="src"                type="input"         port-index="0"               format="BFYX"/>
          <Tensor arg-name="mean"               type="output_buffer" port-index="0" dim="output,0" size="Y*F*4"/>
          <Tensor arg-name="variance"           type="output_buffer" port-index="1" dim="output,0" size="Y*F*4"/>
          <!--other parameters  -->
      </Parameters>
      <WorkSizes dim="output,0" global="((Y+7)/8)*8,F,1" local="8,1,1"/>
  </CustomLayer>
  <CustomLayer name="MVN" stage="1" type="MVCL" version="1">
      <Kernel entry="mvn_scale">
          <Source filename="mvn_scale_changed_orded.bin"/>
      </Kernel>
      <Parameters>
          <Tensor arg-name="src_data"           type="input"        port-index="0"               format="BFYX"/>
          <Tensor arg-name="dst_data"           type="output"       port-index="0"               format="BFYX"/>
          <Tensor arg-name="mean_part"          type="input_buffer" port-index="0" dim="output,0" size="Y*F*4"/>
          <Tensor arg-name="power_mean"         type="input_buffer" port-index="1" dim="output,0" size="Y*F*4"/>
          <!--other parameters  -->
      </Parameters>
      <WorkSizes dim="output,0" global="((Y+7)/8)*8,F,1" local="8,1,1"/>
  </CustomLayer>
  ```
  - Each `Tensor` node that has the type `data` must contain the following attributes:
   - `source` – A name of the blob as it is in the IR. Typical example is `weights` for convolution.
   - `format` – Specifies the channel order in the tensor. Optional conversion layers are generated if the custom layer format is not.
  ```xml
  <CustomLayer name="BinaryConvolution" type="MVCL" version="1">
    <Kernel entry="binary_convolution">
        <Source filename="binary_layers.bin"/>
    </Kernel>
    <Parameters>
        <Tensor arg-name="src_data"      type="input"   port-index="0"                      format="BFYX"/>
        <Data   arg-name="weights_data"  type="data"                     source="weights"   format="ANY"/>
        <Tensor arg-name="dst_data"      type="output"  port-index="0"                      format="BFYX"/>
        <!--other parameters  -->
    </Parameters>
    <WorkSizes dim="output,0" global="X,Y,F" local="1,1,1"/>
  </CustomLayer>
  ```
  - Each `Scalar` node must contain the following attributes:
   - `arg-name` – The name of a kernel parameter in the kernel signature.
   - `type` – `int` or `float` value. It is used for correct argument extraction from IR parameters.
   - `source` – Contains the name of the parameter in the IR file or input/output (`I`/`O`, `In`/`On`, where `n` is a port number)
   followed by dimension `B`(batch), `Y`(height), `X`(width), or `F`(channels).

  - Each `Data` node must contain the following attributes:
    - `arg-name` – The name of a kernel parameter in the kernel signature.
    - `type` – Node type. Currently, `local_data` is the only supported value, which defines buffer allocated in fast local on-chip memory. It is limited to 100KB for all `__local` and
    `__private` arrays defined inside the kernel as well as all `__local` parameters passed to the kernel. Note that a manual-DMA extension requires double buffering.
    If the custom layer is detected to run out of local memory, the inference fails.
    - `dim` – The dim source with the same `direction,port` format used for `WorkSizes` bindings.
    - `size` – Amount of bytes needed. The current expression syntax supports only expression over dimensions of over selected input/output tensor or constants and may be extended in the future.
  The example binding below illustrates a kernel with two local buffers passed to the kernel.
  ```xml
  <CustomLayer name="GRN" type="MVCL" version="1">
      <Kernel entry="grn_NCHW">
          <Source filename="grn.bin"/>
      </Kernel>
      <Parameters>
          <Tensor arg-name="src_data" type="input"         port-index="0"                  format="BFYX"/>
          <Tensor arg-name="dst_data" type="output"        port-index="0"                  format="BFYX"/>
          <Data   arg-name="src"      type="local_data"                      dim="input,0" size="X*F*2" />
          <Data   arg-name="dst"      type="local_data"                      dim="input,0" size="X*F*2" />
          <Scalar arg-name="C"        type="int"           port-index="0"    source="I.F"               />
          <Scalar arg-name="bias"     type="float"                           source="bias"              />
      </Parameters>
      <WorkSizes dim="input,0" global="X,Y,1" local="X,1,1"/>
  </CustomLayer>
```

## Pass Configuration File to OpenVINO™ Runtime

> **NOTE**: If both native and custom layer implementations are present, the custom kernel has a priority over the native one.
Before loading the network that features the custom layers, provide a separate configuration file and load it using the ov::Core::set_property() method with the "CONFIG_KEY" key and the configuration file name as a value before loading the network that uses custom operations to the plugin:

@snippet docs/snippets/vpu/custom_op.cpp part0

## Optimizing Kernels with OpenCL for VPU (Intel® Neural Compute Stick 2)

This section provides optimization guidelines on writing custom layers with OpenCL for VPU devices. Knowledge about general OpenCL
programming model and OpenCL kernel language is assumed and not a subject of this section. The OpenCL model mapping to VPU is described in the table below.

| OpenCL Model  | VPU Mapping|
|-----|----|
| Device code | Executed on SHAVE cores    |
| Private memory | Mapped to CMX internal memory, limited to 100KB per work group, valid only while the work group is executed |
| Local memory   | Mapped to CMX internal memory, limited to 100KB per work group, valid only while the work group is executed |
| Global memory  | Mapped to DDR, used to pass execution preserved parameters for inputs, outputs, and blobs                |
| Work group     | Executed on a single SHAVE core iterating over multiple work items      |

Note that by the OpenCL specification, the work group execution order is not specified. This means that it is your
responsibility to ensure that race conditions among work groups are not introduced. Custom layer runtime spits evenly
work grid among available compute resources and executes them in an arbitrary order. This static scheduling approach works best if the load is evenly spread out across work groups, which is a typical case for Deep Learning kernels. The following guidelines are recommended to use for work group partitioning:

1. Split work evenly across work groups.
2. Adjust work group granularity to maintain equal workload for all compute codes.
3. Set the maximum number of cores using the `max-shaves` attribute for the `CustomLayer` node. This keeps more resources for the rest of topology. It is also useful if the kernel scalability reached its limits, which may happen while optimizing memory bound kernels or kernels with poor parallelization.
4. Try an alternate data layout (`BFXY`/`BYXF`) for the kernel if it improves work group partitioning or data access patterns.
Consider not just specific layer boost, but full topology performance because data conversion layers would be automatically inserted
as appropriate.

Offline OpenCL compiler (`clc`) features automatic vectorization over `get_global_id(0)` usage, if uniform access is detected.
For example, the kernel below could be automatically vectorized:
```cpp
__kernel void cvtf32f16(__global float* restrict inImage, __global half*  restrict outImage,
                        float   scale, float   bais)
{
    int idx = get_global_id(0) + get_global_id(1) * get_global_size(0) + get_global_id(2) * get_global_size(0) * get_global_size(1);
    outImage[idx] = convert_half(inImage[idx]*scale+bais);
}
```
However, this work-group based vectorizer (WGV) conflicts with the default LLVM vectorizer based on superword level parallelism
(SLP) for the current compiler version. Manual vectorization is recommended to provide the best performance for non-uniform code
patterns. WGV works if and only if vector types are not used in the code.

Here is a short list of optimization tips:

1. Help auto-vectorizer ensure non-aliasing pointers for kernel parameters by putting `restrict` where possible.
  - This can give a performance boost, especially for kernels with unrolling, like `ocl_grn` from the example below.
  - Place `restrict` markers for kernels with manually vectorized codes. In the `ocl_grn` kernel below, the unrolled version without `restrict` is up to 20% slower than the most optimal one, which combines unrolling and `restrict`.
2. Put `#&zwj;pragma unroll N` to your loop header. The compiler does not trigger unrolling by default, so it is your responsibility to
annotate the code with pragmas as appropriate. The `ocl_grn` version with `#&zwj;pragma unroll 4` is up to 50% faster, most of which comes from unrolling the first loop, because LLVM, in general, is better in scheduling 3-stage loops (load-compute-store), while the fist loop
 `variance += (float)(src_data[c*H*W + y*W + x] * src_data[c*H*W + y*W + x]);` is only 2-stage (load-compute). Pay
attention to unrolling such cases first. Unrolling factor is loop-dependent. Choose the smallest number that
still improves performance as an optimum between the kernel size and execution speed. For this specific kernel, changing the unroll factor from `4` to `6` results in the same performance, so unrolling factor equal to 4 is an optimum. For Intel® Neural Compute Stick 2, unrolling is conjugated with the automatic software pipelining for load, store, and compute stages:
```cpp
__kernel void ocl_grn(__global const half* restrict src_data, __global half* restrict dst_data, int C, float bias)
{
    int x = get_global_id(0);
    int W = get_global_size(0);
    int y = get_global_id(1);
    int H = get_global_size(1);
    float variance = bias + 1e-9f;
    #pragma unroll 4
    for (int c = 0; c < C; c++)
        variance += (float)(src_data[c*H*W + y*W + x] * src_data[c*H*W + y*W + x]);
    variance = 1.f / native_sqrt(variance);
    #pragma unroll 4
    for (int c = 0; c < C; c++)
        dst_data[c*H*W + y*W + x] = (half)((float)src_data[c*H*W + y*W + x] * variance);
}
```
To check the efficiency of WGV, you can compare performance of the kernel above with the kernel below, which is manually vectorized over width:
```cpp
__kernel void ocl_grn_line(__global const half* restrict src_data,  __global half* restrict dst_data, int C, int W, float bias)
{
    int y   = get_global_id(1);
    int H   = get_global_size(1);
    for (int x = 0; x < W/8; x++)
    {
        float8 variance = (float8)(bias+1e-9f);
        #pragma unroll 4
        for (int c = 0; c < C; c++)
        {
            __global const half8* restrict src_line = ((__global const half8 * restrict)(src_data + c*H*W + y*W));
            half8 sh = src_line[x];
            variance += convert_float8(sh*sh);
        }
        variance = 1.f/native_sqrt(variance);
        #pragma unroll 4
        for (int c = 0; c < C; c++)
        {
            __global const half8* restrict src_line = ((__global const half8 * restrict)(src_data + c*H*W + y*W));
            __global       half8* restrict dst_line = ((__global       half8 * restrict)(dst_data + c*H*W + y*W));
            dst_line[x] = convert_half8(convert_float8(src_line[x])*variance);
        }
    }
    for (int x = W/8*8; x < W; x++)
    {
        float variance = bias+1e-9f;
        #pragma unroll 4
        for (int c = 0; c < C; c++)
            variance += (float)(src_data[c*H*W + y*W + x]*src_data[c*H*W + y*W + x]);
        variance = 1.f/native_sqrt(variance);
        #pragma unroll 4
        for (int c = 0; c < C; c++)
            dst_data[c*H*W + y*W + x] = (float)src_data[c*H*W + y*W + x]*variance;
    }
}
```
Both versions perform the same, but the second one has more complex code.

3. If it is easy to predict the work group size, you can also use the `reqd_work_group_size` kernel attribute to ask the compiler
to unroll the code up to the local size of the work group. Note that if the kernel is actually executed with the
different work group configuration, the result is undefined.

4. Prefer to use the `half` compute if it keeps reasonable accuracy. 16-bit float is a native type for Intel® Neural Compute Stick 2, most of the functions `half_*` are mapped to a single hardware instruction.
Use the standard `native_*` function for the rest of types.

5. Prefer to use the `convert_half` function over `vstore_half` if conversion to 32-bit float is required. `convert_half` is mapped to a single hardware instruction. For the `cvtf32f16` kernel above, the line `outImage[idx] = convert_half(inImage[idx]*scale+bais);` is eight times slower than the code with `vstore_half`.

6. Mind early exits. Early exit can be extremely costly for the current version of the `clc` compiler due to conflicts with the
auto-vectorizer. The generic advice would be to setup local size by `x` dimension equal to inputs or/and outputs width.
If it is impossible to define the work grid that exactly matches inputs or/and outputs to eliminate checks, for example,
`if (get_global_id(0) >= width) return`, use line-wise kernel variant with manual vectorization. 
The kernel example below demonstrates the impact of early exits on kernel performance.
   ```cpp
   // Initial version
   __kernel void reorg(const __global half* restrict src, __global half* restrict out, int stride)
   {
     int w = get_global_id(0);
     int W = get_global_size(0);
     int h = get_global_id(1);
     int H = get_global_size(1);
     int c = get_global_id(2);
     int C = get_global_size(2);
     int C2 = C/(stride*stride);
     int offset = c / C2;
     int c2 = c - C2 * offset;
     int H2 = H*stride;
     int W2 = W*stride;
     int h2 = h*stride + offset / stride;
     int w2 = w*stride + offset - stride * (offset / stride);
     out[W*H*c + W*h + w] = src[W2*H2*c2 + W2*h2 + w2];
   }
   ```
This `reorg` kernel is auto-vectorizable, but an input for YOLO v2 topology is `NCHW=<1,64,26,26>` and it is not multiple of vector width, which is `8` for `half` data type. As a result, the Inference Engine does not select the auto-vectorized kernel.
To compare performance of auto-vectorized and scalar version of the kernel, change the input size to`NCHW=<1,64,26,32>`. This enables the auto-vectorized version to be selected by the Inference Engine and can give you about 30% uplift.
Since the auto-vectorized version is faster, it makes sense to enable it for the YOLO v2 topology input size by setting the local size multiple of vector, for example, 32, and adjust global sizes accordingly. As a result, the execution work grid exceeds actual input dimension, so out-of-bound checks should be inserted. See the updated kernel version below:
   ```cpp
   // Version with out-of-bound checks added
   __kernel void reorg(const __global half* restrict src, __global half* restrict out, int W, int stride)
   {
     int w = get_global_id(0);
     w = min(w, W-1);
     int h = get_global_id(1);
     int H = get_global_size(1);
     int c = get_global_id(2);
     int C = get_global_size(2);
     int C2 = C/(stride*stride);
     int offset = c / C2;
     int c2 = c - C2 * offset;
     int H2 = H*stride;
     int W2 = W*stride;
     int h2 = h*stride + offset / stride;
     int w2 = w*stride + offset - stride * (offset / stride);
     out[W*H*c + W*h + w] = src[W2*H2*c2 + W2*h2 + w2];
   }
   ```
This code performs the same as the initial kernel above (scalar) due to branching overhead. If you replace min/max expression `w = min(w, W-1);` with `if (w >= W) return;`, runtime increases up to 2x against to code without branching (initial version).<br>
If branching is inevitable for your element-based kernel, it is recommended to change the scheme to line-based. See the kernel variant below:
```cpp
// Line-wise version
__kernel void reorg(const __global half* restrict src, __global half* restrict out, int H, int W, int stride)
{
    int h = min((int)get_global_id(0), H-1);
    int c = get_global_id(1);
    int C = get_global_size(1);
    int C2 = C/(stride*stride);
    int offset = c / C2;
    int c2 = c - C2 * offset;
    int H2 = H*stride;
    int W2 = W*stride;
    for (int w = 0; w < W; ++w)
    {
        int h2 = h*stride + offset / stride;
        int w2 = w*stride + offset - stride * (offset / stride);
        out[W*H*c + W*h + w] = src[W2*H2*c2 + W2*h2 + w2];
    }
}
```
This decreases the execution time up to 40% against the best performing vectorized kernel without early exits (initial version).
7. Reuse computations among work items by using line-based kernels or sharing values though `__local` memory.
8. Improve data access locality. Most of custom kernels are memory bound while convolution and fully connected layers are hardware-implemented. The code below demonstrates a further optimized version of the `reorg` kernel unrolled by `stride`:
   ```cpp
   // Unrolled line-wise version
   __kernel void reorg_unrolled_by_stride(const __global half* restrict src, __global half* restrict dst,
                                          int H, int W, int stride)
   {
     int h = min((int)get_global_id(0), H-1);
     int c2 = get_global_id(1);
     int C2 = get_global_size(1);
     int C = C2*stride*stride;
     int H2 = H*stride;
     int W2 = W*stride;
     for (int stride_y = 0; stride_y < stride; stride_y++)
       for (int stride_x = 0; stride_x < stride; stride_x++)
         for (int w2 = 0, w = 0; w < W; w2 += stride, w++)
           dst[W*H*C2*(stride_y*stride+stride_x) + W*H*c2 + W*h + w] = src[W2*H2*c2 + W2*h*stride + W2*stride_y + w2 + stride_x];
   }
   ```
`scr` data in this case loaded only once. As the result, the cycle count drops up to 45% against the line-wise version.

9. Copy data from `__dlobal` to `__local` or `__private` memory if the data is accessed more than once. Access to
`__dlobal` memory is orders of magnitude slower than access to `__local`/`__private` due to statically scheduled pipeline, which
stalls completely on memory access without any prefetch. The same recommendation is applicable for scalar load/store
from/to a `__blobal` pointer since work-group copying could be done in a vector fashion.

10. Use a manual DMA extension. Local (on-chip) memory throughput is up to 24x higher than DDR throughput. Starting from OpenVINO™ 2020.1, VPU OpenCL features manual-DMA kernel extension to copy sub-tensor used by work group into local memory and performing compute without DDR evolved. Here is the simple GRN kernel implementation that runs over DDR. Local size is in the form (width of the input tensor, 1, 1) to define a large enough work group to get code automatically vectorized and unrolled, while global size is (width of the input tensor, height of the input tensor, 1):
   ```cpp
   __kernel void grn_NCHW(
     __global const half* restrict src_data,
     __global       half* restrict dst_data,
     int C,
     float bias)
   {
     float variance = bias + 1e-9f;
     #pragma unroll 4
     for (int c = 0; c < C; c++)
     {
       float val = (float) src_data[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)];
       variance += val*val;
     }
     half hvariance = (half)(native_rsqrt((half)(variance/16.f))*0.25f);
     #pragma unroll 4
     for (int c = 0; c < C; c++)
     {
       dst_data[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)]
       = src_data[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)] * hvariance;
     }
   }
   ```

This kernel can be rewritten to introduce special data binding `__dma_preload` and `__dma_postwrite intrinsics`. This means that instead of one kernel, a group of three kernels should be implemented: `kernelName`, `__dma_preload_kernelName`, and `__dma_postwrite_kernelName`.  `__dma_preload_kernelName` for a particular work group `n` is guaranteed to be executed before the `n`-th work group itself, while `__dma_postwrite_kernelName` is guaranteed to be executed after a corresponding work group. You can define one of those functions that are intended to be used to copy data from-to `__global` and `__local` memory. The syntactics requires exact functional signature match. The example below illustrates how to prepare your kernel for manual-DMA.

   ```cpp
   __kernel void __dma_preload_grn_NCHW(
     __global const half* restrict src,
     __global       half* restrict dst,
     __local        half* restrict local_src,
     __local        half* restrict local_dst,
     int C,
     float bias)
     {
     // ToDO: copy required piece of src tensor into local_src
   }
   
   __kernel void __dma_postwrite_grn_NCHW(
     __global const half* restrict src,
     __global       half* restrict dst,
     __local  const half* restrict local_src,
     __local        half* restrict local_dst,
     int C,
     float bias)
   {
     // ToDO: copy back computed piece of local_dst into dst
   }
   
   __kernel void grn_NCHW(
     __global const half* restrict src_data,
     __global       half* restrict dst_data,
     __local        half* restrict src,
     __local        half* restrict dst,
     int C,
     float bias)
   {
     // same as the example above
   }
   ``` 
The GRN kernel operates on channel-major tensors to compute average over full channel range and then normalizes input elements to produce the output.
As a part of the manual DMA extension, a group of work group copy functions are introduced in addition to `async_work_group_copy`, which is also mapped to a DMA call.

Here is the list of supported functions:
```cpp
// 2D sub-tensor copy
event_t WorkGroupDmaCreateStrideTransaction(
                const local T *src,
                global T *dst,
                size_t  src_width, // width of the line of source in bytes
                size_t  dst_width, // width of the line of destination in bytes
                size_t  src_stride, // stride between corresponding 2 consecutive lines of source in bytes
                size_t  dst_stride, // stride between corresponding 2 consecutive lines of destination in bytes
                size_t size, // total number of bytes loaded for all lines from source to destination
                event_t  event) __OVERLOAD;
event_t WorkGroupDmaCreateStrideTransaction(
                const global T *src,
                local T *dst,
                size_t  src_width, // width of the line of source in bytes
                size_t  dst_width, // width of the line of destination in bytes
                size_t  src_stride, // stride between corresponding 2 consecutive lines of source in bytes
                size_t  dst_stride, // stride between corresponding 2 consecutive lines of destination in bytes
                size_t size, // total number of bytes loaded for all lines from source to destination
                event_t  event) __OVERLOAD;
// 3D sub-tensor copy
event_t WorkGroupDmaCreate3DTransaction(
                 const local T *src,
                 global T *dst,
                 size_t  src_width, // width of the line of source in bytes
                 size_t  dst_width, // width of the line of destination in bytes
                 size_t  src_stride, // stride between corresponding 2 consecutive lines of source in bytes
                 size_t  dst_stride, // stride between corresponding 2 consecutive lines of destination in bytes
                 size_t  num_planes, // number of planes to be copied
                 size_t  src_plane_stride, // stride between corresponding 2 consecutive planes of source in bytes
                 size_t  dst_plane_stride, // stride between corresponding 2 consecutive planes of destination in bytes
                 size_t  size, // size of the loaded plane in bytes, analogues to the size in 2D case
                 event_t  event) __OVERLOAD;
event_t WorkGroupDmaCreate3DTransaction(
                 const global T *src,
                 local T *dst,
                 size_t  src_width, // width of the line of source in bytes
                 size_t  dst_width, // width of the line of destination in bytes
                 size_t  src_stride, // stride between corresponding 2 consecutive lines of source in bytes
                 size_t  dst_stride, // stride between corresponding 2 consecutive lines of destination in bytes
                 size_t  num_planes, // number of planes to be copied
                 size_t  src_plane_stride, // stride between corresponding 2 consecutive planes of source in bytes
                 size_t  dst_plane_stride, // stride between corresponding 2 consecutive planes of destination in bytes
                 size_t  size, // size of the loaded plane in bytes, analogues to the size in 2D case
                 event_t  event) __OVERLOAD;
```
where `T` can be `uchar`, `char`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `half` or `float`.

Modified version of the GRN kernel could be the following:
```cpp
__kernel void __dma_preload_grn_NCHW(
    __global const half* restrict src,
    __global       half* restrict dst,
    __local        half* restrict local_src,
    __local        half* restrict local_dst,
    int C,
    float bias)
{
    WorkGroupDmaCreate3DTransaction(
        src + get_group_id(0)*get_local_size(0)
            + get_group_id(1)*get_local_size(1)*get_global_size(0), // src
        local_src, // dst
        get_local_size(0) * sizeof(half), // src width
        get_local_size(0) * sizeof(half), // dst width
        get_global_size(0) * sizeof(half), // src stride
        get_local_size(0) * sizeof(half), // dst stride
        C, // num planes
        get_global_size(0) * get_global_size(1) * sizeof(half), // src plane stride
        get_local_size(0) * get_local_size(1) * sizeof(half), // dst plane stride
        get_local_size(0) * get_local_size(1) * sizeof(half), // plane size
        0);
}
__kernel void __dma_postwrite_grn_NCHW(
    __global const half* restrict src,
    __global       half* restrict dst,
    __local  const half* restrict local_src,
    __local        half* restrict local_dst,
    int C,
    float bias)
{
    WorkGroupDmaCreate3DTransaction(
        local_dst, // src
        dst + get_group_id(0)*get_local_size(0)
            + get_group_id(1)*get_local_size(1)*get_global_size(0), // dst
        get_local_size(0) * sizeof(half), // src width
        get_local_size(0) * sizeof(half), // dst width
        get_local_size(0) * sizeof(half), // src stride
        get_global_size(0) * sizeof(half), // dst stride
        C, // num planes
        get_local_size(0) * get_local_size(1) * sizeof(half), // src plane stride
        get_global_size(0) * get_global_size(1) * sizeof(half), // dst plane stride
        get_local_size(0) * get_local_size(1) * sizeof(half), // plane size
        0);
}
__kernel void grn_NCHW(
    __global const half* restrict src_data,
    __global       half* restrict dst_data,
    __local        half* restrict src,
    __local        half* restrict dst,
    int C,
    float bias)
{
    float variance = bias + 1e-9f;
    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        float val = (float) src[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)];
        variance += val*val;
    }
    half hvariance = (half)(native_rsqrt((half)(variance/16.f))*0.25f);
    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        dst[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)]
        = src[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)] * hvariance;
    }
}
```

Note the `get_local_size` and `get_local_id` usage inside the kernel. 21x speedup is expected for a kernel on enet-curbs setup because it was completely limited by memory usage.

An alternative method to using DMA is to use work item copy extension. Those functions are executed inside a kernel and requires work groups equal to single work item.

Here is the list of supported work item functions:
```cpp
item_dma_event_t WorkItemDmaCreateTransaction(
            const global T *src,
            private T *dst,
            size_t  size,
            item_dma_event_t  event) __OVERLOAD;
item_dma_event_t WorkItemDmaCreateTransaction(
            const private T *src,
            global T *dst,
            size_t  size,
            item_dma_event_t  event) __OVERLOAD;
item_dma_event_t WorkItemDmaCreateStrideTransaction(
                const global T *src,
                private T *dst,
                size_t  src_width,
                size_t  dst_width,
                size_t  src_stride,
                size_t  dst_stride,
                size_t size,
                item_dma_event_t  event) __OVERLOAD;
item_dma_event_t WorkItemDmaCreateStrideTransaction(
                const private T *src,
                global T *dst,
                size_t  src_width,
                size_t  dst_width,
                size_t  src_stride,
                size_t  dst_stride,
                size_t size,
                item_dma_event_t  event) __OVERLOAD;
item_dma_event_t WorkItemDmaCreate3DTransaction(
                const global T *src,
                private T *dst,
                size_t  src_width,
                size_t  dst_width,
                size_t  src_stride,
                size_t  dst_stride,
                size_t  num_planes,
                size_t  src_plane_stride,
                size_t  dst_plane_stride,
                size_t  size,
                item_dma_event_t  event) __OVERLOAD;
item_dma_event_t WorkItemDmaCreate3DTransaction(
                const private T *src,
                global T *dst,
                size_t  src_width,
                size_t  dst_width,
                size_t  src_stride,
                size_t  dst_stride,
                size_t  num_planes,
                size_t  src_plane_stride,
                size_t  dst_plane_stride,
                size_t  size,
                item_dma_event_t  event) __OVERLOAD;
```
where `T` can be `uchar`, `char`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `half` or `float`.
