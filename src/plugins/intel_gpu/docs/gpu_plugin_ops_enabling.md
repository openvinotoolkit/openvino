# GPU plugin operations enabling flow

## Terminology
* **NGraph operation**: Building block of neural networks, such as convolution or pooling.
* **(clDNN) Primitive**: Basic NN operation that was defined in clDNN. One primitive is usually mapped to one ngraph operation, but graph compilation may cause the mapping not to be 1-to-1.
* **Kernel**: Actual body of execution in GPU. It also refers to specific implementations of **Primitive** for GPU, such as `convolution_gpu_winograd_2x3_s1.cl`. Usually, single kernel fulfills the operation of single primitive, but several kernels may be used to support one primitive.
* **Unittest**: Single-layer test within cldnn.
* **Functional test**: Single-layer test in IE.

<br>

## Adding new primitive
1. Understand the new operation.
    * Review the [ngraph operation spec](https://github.com/openvinotoolkit/openvino/tree/master/docs/ops)
    * IE operations(a.k.a primitive or NN-layer) are defined by ngraph.
    * You can check ngraph reference implementation of the primitive as well
        * e.g. [Scatter Elements Update in nGraph](https://github.com/openvinotoolkit/openvino/blob/master/src/core/reference/include/ngraph/runtime/reference/scatter_elements_update.hpp)

1. Try to find existing primitive that fully or partially covers this operation.
    * It is also possible to transform the network so that the missing primitive is covered from existing primitive.
    * e.g. [Replace reduce with pooling](https://github.com/openvinotoolkit/openvino/blob/23808f46f7b5d464fd649ad278f253eec12721b3/inference-engine/src/cldnn_engine/cldnn_engine.cpp#L205)

1. Add new / extend existing cldnn primitive according to the operation spec.
    1. This phase is to enable primitive within cldnn library, without exposing it to IE.
    1. Implement **reference parallel kernel** that supports all parameters of the operation and all input/output data types and layouts
        
        | File | Description |
        |------|-------------|
        | [scatter_elements_update_ref.cl](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/scatter_elements_update_ref.cl) | OpenCL Kernel body. For more detail, please see [How to write OCL kernel](#writing-ocl-kernel) section |
        | [scatter_elements_update_kernel_ref.(cpp,h)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/kernel_selector/kernels/scatter_update/scatter_elements_update_kernel_ref.cpp) | Counterpart of kernel body for host |
        | [scatter_elements_update_kernel_selector.(cpp,h)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/kernel_selector/kernels/scatter_update/scatter_elements_update_kernel_selector.cpp) | Kernel selector for a primitive |
        | [register_gpu.(cpp,hpp)](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/thirdparty/clDNN/src/gpu/register_gpu.cpp) | Primitive registration |
        | [scatter_elements_update_gpu.cpp](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/thirdparty/clDNN/src/gpu/scatter_elements_update_gpu.cpp) | Primitive registration, input spec |
        | [scatter_elements_update_inst.h](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/graph/include/scatter_elements_update_inst.h) | Node type declaration for cldnn program |
        | [clDNN/src/scatter_elements_update.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/graph/scatter_elements_update.cpp) | Code for scatter_elements_update_inst.h |
        | [clDNN/api/cldnn/primitives/scatter_elements_update.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/include/intel_gpu/primitives/scatter_elements_update.hpp) | clDNN primitive definition |
        | [common_types.h](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/kernel_selector/common_types.h) | Enum declaration for KernelType and arguments |

    1. Add unit tests for the new operation

        | File | Description |
        |------|-------------|
        | [scatter_elements_update_gpu_test.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/test_cases/scatter_elements_update_gpu_test.cpp) | Unittest for layer |

        * Need to add reference code or expected result for checking the result.

        * You can also specify the kernel with `force_implementations` in case the primitive contains multiple kernels.
            ```
            ...
            build_options options;
            implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
            options.set_option(build_option::force_implementations({ {"conv_fsv", conv_impl} }));
            network network(engine, topology, options);
            ...
            ```

        * This unit test is built into `clDNN_unit_tests`. It is a gtest application.
            ```
            # Show list of test cases
            openvino/bin/intel64/Debug$ ./clDNN_unit_tests64 --gtest_list_tests
            # Run test
            openvino/bin/intel64/Debug$ ./clDNN_unit_tests64 --gtest_filter=scatter_elements_update_gpu_fp16.*
            ```
        
        * Test scope needs to be comprehensive, but not wasteful. These tests run for every PRs in CI. Let's save the planet.
    
    1. Support layer fusion, if applicable
        * It is usually easy to fuse some layers, such as scale, activation, quantize and eltwise, into previous layer. This fusing rule can be added to `prepare_primitive_fusing::fuse_simple_primitives`.
        * `fuse_simple_primitives` is called during [graph compilation phase](https://github.com/openvinotoolkit/openvino/blob/71c50c224964bf8c24378d16f015d74e2c1e1ce8/inference-engine/thirdparty/clDNN/src/program.cpp#L430)
        * You can see general description of layer fusion [here](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_CL_DNN.html#optimizations)
        * Unit tests for layer fusion are placed in a single file: [fusings_gpu_test.cpp](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/thirdparty/clDNN/tests/test_cases/fusings_gpu_test.cpp). It is also compiled into `clDNN_unit_tests`.
        * Code for fused layers are generated with `jitter`. It is created as `FUSED_OPS..` macro in OCL code. This generation logic is in `KernelBase::MakeFusedOpsJitConstants`.

1. Add / update factory for this operation in the GPU plugin to use new primitive in inference-engine

    | File | Description |
    |------|-------------|
    | [cldnn_engine/ops/scatter_elements_update.cpp](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/src/cldnn_engine/ops/scatter_elements_update.cpp) | Instantiation from cldnn plugin for IE |
    | [cldnn_primitives_list.hpp](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/src/cldnn_engine/cldnn_primitives_list.hpp) | Registration for primitives |

1. Add functional single layer tests for the operation and try to cover most of the difference use cases of this operation

    | File | Description |
    |------|-------------|
    | [single_layer_tests/scatter_elements_update.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/tests/functional/shared_tests_instances/single_layer_tests/scatter_elements_update.cpp) | Single layer test |

    * It is possible to use ngraph reference code for result validation.
    * This is compiled into `gpuFuncTests`. It is also `gtest` application.
    * Please also review the [general guideline of test infrastructure](https://github.com/openvinotoolkit/openvino/wiki/InferenceEngineTestsInfrastructure)

1. [Optional] If there are existing IRs with this operation, try to run the full model(s) to be sure that it's correctly processed within the context

1. [Optional] If there are existing IRs with this operation, try to run the full model(s) and estimate performance impact from this operation on total model execution time

1. Create PR with your changes
    * If you are `OpenVINO` group member in github, CI will be triggered.
    * Please review the [OpenVINO contribution guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md).

<br>

## Adding new kernel for an existing primitive
* The process is quite similar to previous one. You can skip already existing steps.
* Main work is adding new kernel and registering it from kernel selector.
* You may need to add unit test for that new kernel. Specific kernel can be chosen with `build_option::force_implementations`.
* It is not possible to specify kernel from functional test(IE).

<br>

## Writing OCL kernel

### Jitter
In GPU OCL kernels, many conditional statements are processed with `#ifdef` so that it can be handled during compile-time. The definitions are created with `jitter.cpp`. It is set during graph compilation. You can see generated macros following the steps in [source dumps](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/gpu_debug_utils.md#sources-dumps).
Jitter also contains run-time parameters such as input and output size.
Additional macros can be defined from host-code of kernel itself. For example, see below code snippet. It passes `SUB_GROUP_SIZE` through macro definition through jitter.
```
  // GetJitConstants method of the kernel
  const size_t sub_group_size = 16;
  JitConstants jit = MakeBaseParamsJitConstants(params);
  jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size ));
```

### Accessing input and output tensor
Jitter generates macros for index calculations. With these macros, you can program ocl kernel in a layout-agnostic way. If you use the macro `${TENSOR_NAME}_GET_INDEX`, you can get 1d-index from tensor coordinate whether the format is planar(such as `bfyx` or `byxf`) or blocked.(such as `b_fs_yx_fsv16`). You can check [source code for GET_INDEX macro](https://github.com/openvinotoolkit/openvino/blob/7f8d3aa63899a3e3362c95eb7d1b04a5899660bd/inference-engine/thirdparty/clDNN/kernel_selector/core/common/jitter.cpp#L313).

### Layout support
If a kernel is not performance-critical, you can support `bfyx`, `bfzyx` and `bfwzyx` only for layout. Those are default layouts. As an optimized format, `b_fs_yx_fsv16`, `b_fs_yx_fsv4` or `byxf` can be used as well.
[General description of layout can be found here](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/gpu_memory_formats.md) and [header file is here](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/thirdparty/clDNN/api/tensor.hpp)

### Layer fusion
When layers are fused, `jitter` will create macros to generate code for fused layers. It is realized into `FUSED_OPS..` in OCL kernel. You can understand the usage from other kernels.
There is a [comment that describes layer fusion](https://github.com/openvinotoolkit/openvino/blob/7f8d3aa63899a3e3362c95eb7d1b04a5899660bd/inference-engine/thirdparty/clDNN/kernel_selector/core/kernel_selector_params.h#L521).

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)