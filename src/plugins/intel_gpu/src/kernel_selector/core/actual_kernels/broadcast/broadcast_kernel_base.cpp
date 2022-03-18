// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants BroadcastKernelBase::GetJitConstants(const broadcast_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("BROADCAST_ORDER", params.input_order)});

    return jit;
}

BroadcastKernelBase::DispatchData BroadcastKernelBase::SetDefault(const broadcast_params& params) {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                     { Tensor::DataChannelName::Y, Tensor::DataChannelName::Z, Tensor::DataChannelName::W },
                                                                     { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

    dispatchData.gws = { output.X().v, output.Y().v * output.Z().v * output.W().v, output.Batch().v * output.Feature().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsData BroadcastKernelBase::GetCommonKernelsData(const Params& params,
                                                      const optional_params& options) const {
    assert(params.GetType() == KernelType::BROADCAST);

    const auto& prim_params =
        static_cast<const broadcast_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<broadcast_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {k_data};
}
}  // namespace kernel_selector
