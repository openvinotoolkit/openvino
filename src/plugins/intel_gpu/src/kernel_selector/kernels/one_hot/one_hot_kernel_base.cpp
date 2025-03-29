// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants OneHotKernelBase::GetJitConstants(const one_hot_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
         MakeJitConstant("ONE_HOT_AXIS", params.one_hot_axis),
         MakeJitConstant("ONE_HOT_LIMIT", params.one_hot_limit),
         MakeJitConstant("ON_VALUE", params.on_value),
         MakeJitConstant("OFF_VALUE", params.off_value)
    });

    return jit;
}

OneHotKernelBase::DispatchData OneHotKernelBase::SetDefault(const one_hot_params& params) {
    const auto& input = params.inputs[0];
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    DispatchData dispatchData;
    if (params.outputs[0].GetDims().size() == 5) {
        dispatchData.gws = { input.Batch().v, input.Feature().v * input.Z().v, input.Y().v * input.X().v };
        dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                       { Tensor::DataChannelName::Z, Tensor::DataChannelName::FEATURE },
                       { Tensor::DataChannelName::X, Tensor::DataChannelName::Y }};
    } else {
        dispatchData.gws = { input.Batch().v, input.Feature().v, input.Y().v * input.X().v };
        dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                       { Tensor::DataChannelName::FEATURE },
                       { Tensor::DataChannelName::X, Tensor::DataChannelName::Y }};
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsData OneHotKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::ONE_HOT);

    const auto& prim_params =
        static_cast<const one_hot_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<one_hot_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {k_data};
}
}  // namespace kernel_selector
