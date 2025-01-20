// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_kernel_opt.h"

namespace kernel_selector {
ParamsKey ArgMaxMinKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey ArgMaxMinKernelOpt::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_reqd_subgroup_size();
    k.requires_subgroups();
    k.requires_subgroup_shuffle();
    k.requires_subgroup_shuffle_relative();

    return k;
}

KernelsData ArgMaxMinKernelOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const arg_max_min_params& orgParams = static_cast<const arg_max_min_params&>(params);

    size_t topK = orgParams.topK;
    size_t size = (size_t)(orgParams.inputs[0].X().v * orgParams.inputs[0].Y().v * orgParams.inputs[0].Feature().v) / 8;
    size_t outSize = size / 16 * topK;
    int kernelAmount = 1;
    for (; outSize > 128; outSize = (size_t)((outSize / 128 + 1) * topK)) {
        kernelAmount++;
    }
    KernelData kd = KernelData::Default<arg_max_min_params>(params, kernelAmount);
    for (int i = 0; i < kernelAmount; i++) {
        DataTensor input;
        if (i == 0)
            input = orgParams.inputs[0];
        else
            input = orgParams.outputs[0];

        auto newParams = orgParams;
        newParams.inputs.resize(1);
        newParams.inputs[0] = input;

        auto& kernel = kd.kernels[i];
        DispatchData dispatchData = SetDefault(newParams);
        auto cldnnJit = GetJitConstants(newParams);
        auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, params);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        dispatchData.gws = { Align(size, 16), orgParams.inputs[0].Batch().v, 1 };
        dispatchData.lws = { 16, 1, 1 };

        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entryPoint);
        size = (size / 128 + 1) * topK;
    }

    return {kd};
}

KernelsPriority ArgMaxMinKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
