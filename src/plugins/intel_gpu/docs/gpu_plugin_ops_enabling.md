# GPU Plugin Operations Enabling Flow

## Terminology

* **NGraph operation**: Building block of neural networks, such as convolution or pooling.
* **(GPU) Primitive**: Basic NN operation that was defined in GPU. One primitive is usually mapped to one ngraph operation, but graph compilation may cause the mapping not to be 1-to-1.
* **Kernel**: Actual body of execution in GPU. It also refers to specific implementations of **Primitive** for GPU, such as `convolution_gpu_winograd_2x3_s1.cl`. Usually, single kernel fulfills the operation of a single primitive, but several kernels may be used to support one primitive.
* **Unittest**: Single-layer test within GPU plugin.
* **Functional test**: Single-layer test in IE.

## Adding new primitive

1. Understand the new operation.
    * Review the [ngraph operation spec](https://github.com/openvinotoolkit/openvino/tree/master/docs/ops)
    * IE operations(a.k.a primitive or NN-layer) are defined by ngraph.
    * You can check ngraph reference implementation of the primitive as well
        * For example, [Scatter Elements Update in nGraph](https://github.com/openvinotoolkit/openvino/blob/master/src/core/reference/include/openvino/reference/scatter_elements_update.hpp)

1. Try to find existing primitive that fully or partially covers this operation.
    * It is also possible to transform the network so that the missing primitive is covered from existing primitive.
    * For example, [replace reduce with pooling](https://github.com/openvinotoolkit/openvino/blob/23808f46f7b5d464fd649ad278f253eec12721b3/inference-engine/src/cldnn_engine/cldnn_engine.cpp#L205).

1. Add new / extend existing GPU primitive according to the operation spec.
    1. This phase is to enable primitive within GPU plugin, without exposing it to IE.
    1. Implement **reference parallel kernel** that supports all parameters of the operation and all input/output data types and layouts.

        | File | Description |
        |------|-------------|
        | [scatter_elements_update_ref.cl](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/scatter_elements_update_ref.cl) | OpenCL Kernel body. For more detail, please see [How to write OCL kernel](#writing-ocl-kernel) section |
        | [scatter_elements_update_kernel_ref.(cpp,h)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/kernel_selector/kernels/scatter_update/scatter_elements_update_kernel_ref.cpp) | Counterpart of kernel body for host |
        | [scatter_elements_update_kernel_selector.(cpp,h)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/kernel_selector/kernels/scatter_update/scatter_elements_update_kernel_selector.cpp) | Kernel selector for a primitive |
        | [register.(cpp,hpp)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/graph/impls/ocl/register.cpp) | Primitive registration |
        | [scatter_elements_update.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/graph/impls/ocl/scatter_elements_update.cpp) | Primitive registration, input spec |
        | [scatter_elements_update_inst.h](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/graph/include/scatter_elements_update_inst.h) | Node type declaration for GPU program |
        | [src/graph/scatter_elements_update.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/graph/scatter_elements_update.cpp) | Code for scatter_elements_update_inst.h |
        | [primitives/scatter_elements_update.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/include/intel_gpu/primitives/scatter_elements_update.hpp) | GPU primitive definition |
        | [common_types.h](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/kernel_selector/common_types.h) | Enum declaration for KernelType and arguments |

    1. Add unit tests for the new operation.

        | File | Description |
        |------|-------------|
        | [scatter_elements_update_gpu_test.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/unit/test_cases/scatter_elements_update_gpu_test.cpp) | Unittest for layer |

        * You need to add reference code or expected result for checking the result.

        * You can also specify the kernel with `force_implementations` in case the primitive contains multiple kernels.
            ```
            ...
            build_options options;
            implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
            options.set_option(build_option::force_implementations({ {"conv_fsv", conv_impl} }));
            network network(engine, topology, options);
            ...
            ```

        * This unit test is built into `ov_gpu_unit_tests`. It is a `gtest` application.
            ```
            # Show list of test cases
            openvino/bin/intel64/Debug$ ./ov_gpu_unit_tests --gtest_list_tests
            # Run test
            openvino/bin/intel64/Debug$ ./ov_gpu_unit_tests --gtest_filter=scatter_elements_update_gpu_fp16.*
            ```

        * Test scope needs to be comprehensive, but not wasteful. These tests run for every PR in CI. Let's save the planet.

    1. Support layer fusion, if applicable
        * It is usually easy to fuse some layers, such as *scale*, *activation*, *quantize*, and *eltwise*, into the previous layer. This fusing rule can be added to `prepare_primitive_fusing::fuse_simple_primitives`.
        * `fuse_simple_primitives` is called during [graph compilation phase](https://github.com/openvinotoolkit/openvino/blob/71c50c224964bf8c24378d16f015d74e2c1e1ce8/inference-engine/thirdparty/clDNN/src/program.cpp#L430)
        * Unit tests for layer fusion: [link](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/unit/fusions/scatter_elements_update_fusion_test.cpp). It is also compiled into `ov_gpu_unit_tests`.
        * Code for fused layers are generated with `jitter`. It is created as `FUSED_OPS..` macro in OCL code. This generation logic is in `KernelBase::MakeFusedOpsJitConstants`.

1. Add / update factory for this operation in the GPU plugin to use new primitive in inference-engine.

    | File | Description |
    |------|-------------|
    | [plugin/ops/scatter_elements_update.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/plugin/ops/scatter_elements_update.cpp) | Instantiation of gpu plugin primitive from IE |
    | [primitives_list.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/include/intel_gpu/plugin/primitives_list.hpp) | Registration for primitives |

1. Add functional single-layer tests for the operation and try to cover most of the different use cases of this operation.

    | File | Description |
    |------|-------------|
    | [single_layer_tests/scatter_elements_update.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/tests/functional/shared_tests_instances/single_layer_tests/scatter_elements_update.cpp) | Single layer test |

    * It is possible to use ngraph reference code for result validation.
    * This is compiled into `gpuFuncTests`. It is also `gtest` application.
    * Also, review the [general guideline of test infrastructure](https://github.com/openvinotoolkit/openvino/blob/master/docs/IE_PLUGIN_DG/PluginTesting.md).

1. [Optional] If there are existing IRs with this operation, try to run the full model(s) to be sure that it is correctly processed within the context.

1. [Optional] If there are existing IRs with this operation, try to run the full model(s) and estimate performance impact from this operation on total model execution time.

1. Create a PR with your changes.
    * If you are `OpenVINO` group member in github, CI will be triggered.
    * Review the [OpenVINO contribution guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md).

## Adding new kernel for an existing primitive

* The process is quite similar to the previous one. You can skip already existing steps.
* Main work is adding a new kernel and registering it from the kernel selector.
* You may need to add a unit test for that new kernel. A specific kernel can be chosen with `build_option::force_implementations`.
* It is not possible to specify a kernel from a functional test(IE).

## Writing OCL kernel

### Jitter

In GPU OCL kernels, many conditional statements are processed with `#ifdef` so that they can be handled during compile-time. The definitions are created with `jitter.cpp`. It is set during graph compilation. You can see generated macros, following the steps in [source dumps](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/gpu_debug_utils.md#sources-dumps).

Jitter also contains run-time parameters such as input and output size.
Additional macros can be defined from the host-code of a kernel itself. For example, see the code snippet below. It passes `SUB_GROUP_SIZE` through macro definition through jitter.
```
  // GetJitConstants method of the kernel
  const size_t sub_group_size = 16;
  JitConstants jit = MakeBaseParamsJitConstants(params);
  jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size ));
```

### Accessing input and output tensor

Jitter generates macros for index calculations. With these macros, you can program OCL kernel in a layout-agnostic way. If you use the macro `${TENSOR_NAME}_GET_INDEX`, you can get 1d-index from a tensor coordinate whether the format is planar (such as `bfyx` or `byxf`) or blocked (such as `b_fs_yx_fsv16`). You can check [source code for GET_INDEX macro](https://github.com/openvinotoolkit/openvino/blob/7f8d3aa63899a3e3362c95eb7d1b04a5899660bd/inference-engine/thirdparty/clDNN/kernel_selector/core/common/jitter.cpp#L313).

### Layout support

If a kernel is not performance-critical, you can support `bfyx`, `bfzyx` and `bfwzyx` only for layout. Those are default layouts. As an optimized format, `b_fs_yx_fsv16`, `b_fs_yx_fsv4` or `byxf` can be used as well.

[General description of layout can be found here](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/gpu_memory_formats.md) and [header file is here](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/include/intel_gpu/runtime/format.hpp).

### Layer fusion

When layers are fused, `jitter` will create macros to generate code for fused layers. It is realized into `FUSED_OPS..` in OCL kernel. You can understand the usage from other kernels.
There is a [comment that describes layer fusion](https://github.com/openvinotoolkit/openvino/blob/7f8d3aa63899a3e3362c95eb7d1b04a5899660bd/inference-engine/thirdparty/clDNN/kernel_selector/core/kernel_selector_params.h#L521).

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)