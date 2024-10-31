// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "search_sorted_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants SearchSortedKernelBase::GetJitConstants(const search_sorted_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    // jit.AddConstants({
    //      MakeJitConstant("search_sorted_AXIS", params.search_sorted_axis),
    //      MakeJitConstant("search_sorted_LIMIT", params.search_sorted_limit),
    //      MakeJitConstant("ON_VALUE", params.on_value),
    //      MakeJitConstant("OFF_VALUE", params.off_value)
    // });

    return jit;
}

SearchSortedKernelBase::DispatchData SearchSortedKernelBase::SetDefault(const search_sorted_params& params) {
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

KernelsData SearchSortedKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::SEARCH_SORTED);

    const auto& prim_params =
        static_cast<const search_sorted_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<search_sorted_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {k_data};
}
}  // namespace kernel_selector
