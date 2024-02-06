# Dynamic primitive impls

* Unlike static impl, dynamic impl has no shape information before execution. The input shape is dependent with input layer's output shape or execution result value. To do handle this dynamic case, the shape agnostic kernel is used.

* Kernels that support dynamic shapes must declare EnableDynamicShapesSupport() as supported key.

```cpp
ParamsKey CumSumKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableDynamicShapesSupport();
    return k;
}
```

* Just same as static shape, dynamic impl for each primitive should be added in 'implementation_map' with impl_types::ocl for GPU with supported types and formats

```cpp
implementation_map<activation>::add(impl_types::ocl,
                                    shape_types::dynamic_shape,
                                    typed_primitive_impl_ocl<activation>::create<activation_impl>,
                                    types,
                                    dyn_formats);

implementation_map<activation>::add(impl_types::ocl,
                                    shape_types::static_shape,
                                    typed_primitive_impl_ocl<activation>::create<activation_impl>,
                                    keys);
```

## Shape agnostic kernel 

The shape agnostic kernel is an implementation which can work for variable shape instead of predefined shapes. It declares some variables with dim information of input and output inside the kernel and is replaced with the required value at each execution.

#### Update shape information
Inputs and outputs shape information of the kernel should be updated for every execution.

* The __Shape information__ to be used inside the kernel is stored as **memory::ptr _shape_info_memory** in __class primitive_inst__.

* This shape information is determined in **update_shape_info** during **update_impl()**.

* In **update_shape_info**, all shapes and padding values of dynamic inputs and all shapes and padding values of dynamic outputs are stored in order.

#### OPTIONAL_SHAPE_INFO_ARG
Define OPTIONAL_SHAPE_INFO_ARG and OPTIONAL_SHAPE_INFO_TENSOR to be used in shape agnostic kernel

> openvino/src/plugins/intel_gpu/src/kernel_selector/kernel_base.cpp
```cpp
if (params.is_shape_agnostic) {
    jit.AddConstant(MakeJitConstant("IS_DYNAMIC", 1));
    jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_ARG", "__global const int* shape_info,"));
    jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_TENSOR", "shape_info,"));
} else {
    jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_ARG", ""));
    jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_TENSOR", ""));
}
```

#### Update dispatch data
**update_dispatch_data_func** should be defined for every shape-agnostic kernel impl. This is called for every execution. Global workgroup, local workgroup, and skip_execution should be updated. And more Kernel data could be updated in here if necessary for every execution.

```cpp
kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
    const auto& prim_params = static_cast<const activation_params&>(params);
    auto dispatchData = SetDefault(prim_params);
    OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
    kd.kernels[0].params.workGroups.global = dispatchData.gws;
    kd.kernels[0].params.workGroups.local = dispatchData.lws;
    kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
};
```

#### Example to use shape_info[]

* Example of **shape_info[]** in kernel
```cpp  
KERNEL(shape_of_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global OUTPUT_TYPE* output
    )
{
    const unsigned int i = (uint)get_global_id(2);

#if IS_DYNAMIC
    output[i] = TO_OUTPUT_TYPE(shape_info[i]);  // shape_info[] is directly used
#else
    size_t shapes[] = INPUT_DIMS_INIT;
    output[i] = TO_OUTPUT_TYPE(shapes[i]);
#endif
}
```