// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "istft_kernel_base.h"

#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants ISTFTKernelBase::GetJitConstants(const ISTFT_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("CENTER", params.center)});
    jit.AddConstants({MakeJitConstant("NORMALIZED", params.normalized)});
    jit.AddConstants({MakeJitConstant("LENGTH_BUFFER", params.inputs.size() == 5)});

    return jit;
}

void ISTFTKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const ISTFT_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

ISTFTKernelBase::DispatchData ISTFTKernelBase::SetDefault(const ISTFT_params& params) {
    CommonDispatchData dispatchData;
    const auto inLayout = params.inputs.front().GetLayout();
    const auto input0 = params.inputs.front();
    const auto input1 = params.inputs[1];
    const auto& output = params.outputs.front();
    const auto outLayout = output.GetLayout();

    OPENVINO_ASSERT(output.Dimentions() == 4);

    std::vector<std::vector<Tensor::DataChannelName>> dimsByGws;
    dispatchData.gws = {input0.Batch().v, input0.Y().v, input1.X().v};
    dimsByGws = {{Tensor::DataChannelName::BATCH}, {Tensor::DataChannelName::Y}, {Tensor::DataChannelName::X}};

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, inLayout, outLayout, dimsByGws);

    return dispatchData;
}

KernelsData ISTFTKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::ISTFT);

    const auto& prim_params = static_cast<const ISTFT_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<ISTFT_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(k_data);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     static_cast<int>(prim_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     prim_params.is_shape_agnostic);

    return {k_data};
}
}  // namespace kernel_selector
