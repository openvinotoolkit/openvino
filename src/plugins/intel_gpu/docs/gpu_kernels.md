# GPU Kernels Implementation Overview

As mentioned in [GPU plugin structure](./source_code_structure.md), kernels for GPU plugin are located in `src/plugins/intel_gpu/src/kernel_selector` folder.

For each operation, there are usually multiple kernels that can support different parameters and/or are optimized for different scenarios.

Each operation has 3 major entities in kernel selector:
 - Operation specific `kernel_selector` instance
 - Operation parameters descriptor
 - Kernels itself with a set of heuristics inside for optimal selection

## Kernel selector instance

For each operation, you create `kernel_selector` class derived from `kernel_selector_base`. Basically, this class is needed to specify available kernels
for a given operation. Each kernel selector is used as a singleton. For example:

```cpp
class mvn_kernel_selector : public kernel_selector_base {
public:
    static mvn_kernel_selector& Instance() {
        static mvn_kernel_selector instance_;
        return instance_;
    }

    mvn_kernel_selector();

    KernelsData GetBestKernels(const Params& params) const override;
}

// The list of available kernels is usually specified in kernel_selector c-tor using `Attach` method whith creates instance of each type
// and append it to implementations list.
// In this case we have 3 available kernels for MVN operation. Kernels might have different priorities and support only subset of operation parameters
// E.g. MVNKernel_b_fs_yx_fsv16_imad supports only `fsv16` blocked layouts and INT8/UINT8 input data types
mvn_kernel_selector::mvn_kernel_selector() {
    Attach<MVNKernelRef>();
    Attach<MVNKernelBfyxOpt>();
    Attach<MVNKernel_b_fs_yx_fsv16_imad>();
}

// This method is used to get the optimal kernel for given parameters
// There are 2 base methods to pick optimal kernels: `GetNaiveBestKernel` and `GetAutoTuneBestKernel`
// If kernel supports auto tuning, then it uses `GetAutoTuneBestKernel`, otherwise, it uses `GetNaiveBestKernel`
// parameterized with `KernelType` which specifies the operation type which is implemented by the specific kernel selector
KernelsData mvn_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, options, KernelType::MVN);
}
```

The caller code looks as follows:

```cpp
// Get static instance of the kernel_selector
auto& kernel_selector = kernel_selector::mvn_kernel_selector::Instance();
// Run some heuristics to pick the best mvn kernel for given `mvn_params`
auto best_kernels = kernel_selector.GetBestKernels(mvn_params);
```

## Operation parameters

The parameters of operation for `kernel_selector` are defined in corresponding `${op_name}_params` class which is derived from `base_params`. For example:
```cpp
struct mvn_params : public base_params {
    mvn_params() : base_params(KernelType::MVN) {}

    MVNMode mvnMode = MVNMode::WITHIN_CHANNELS;
    bool mvnNormalizeVariance = true;
    float epsilon = 1e-10f;

    virtual ParamsKey GetParamsKey() const {
        ParamsKey k = base_params::GetParamsKey();

        k.EnableMVNMode(mvnMode);

        if (mvnNormalizeVariance)
            k.EnableMVNNormalizeVariance();

        return k;
    }
};
```

The derived class should parameterize base class with a specific `KernelType` and add operation-specific parameters. The only method that must be implemented
is `GetParamsKey()` which is used as a quick check for kernels applicability for current parameters. In other words, you take a `ParamsKey` object calculated for input
operation parameters and a `ParamsKey` object for each kernel. Then, you can compare them and discard the kernels that do not support current parameters.
`ParamsKey` is implemented as a set of bit masks, so the applicability check is quite simple:
```cpp
const ParamsKey implKey = some_implementation->GetSupportedKey();
if (!implKey.Support(paramsKey))
    // Do something

// Support() method do something like follows for each internal bit mask:
if (!((implKey.mask & paramsKey.mask) == paramsKey.mask))
    return false;
```

## Kernel implementation

Each kernel must specify the following things:
- Input parameters checks
  - `GetSupportedKey()` method implementation, which returns `ParamsKey` object for current implementation.
  - `Validate()` method, that does more complex checks (optional).
- Dispatch data (global/local workgroup sizes, scheduling algorithm, etc.)
- Kernel name - must be passes to base class c-tor
- Kernel arguments specification - description of each argument in corresponding OpenCL™ kernel
- Additional JIT constants required for kernel - set of macro definitions that must be added to the kernel template to make full specialization for given params
- Supported fused operations (if any) - a list of supported operations that can be fused into the current kernel.

Key methods of each kernel implementation are as follows:

```cpp
class MVNKernelRef : public MVNKernelBase {
public:
    MVNKernelRef() : MVNKernelBase("mvn_gpu_ref") {} // mvn_gpu_ref is the name of the file with kernel template in cl_kernels/ folder without .cl extension
    // Returns the kernel specified for input parameters if the implementation can process it
    KernelsData GetKernelsData(const Params& params) const override;
    // Returns `ParamsKey` for current implementation for quick applicability check
    ParamsKey GetSupportedKey() const override;

protected:
    // Specifies additional jit constants for kernel template specification
    JitConstants GetJitConstants(const mvn_params& params, DispatchData dispatchData) const override;
    // The list of supported fused operations
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE
        };
    }
};
```

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
