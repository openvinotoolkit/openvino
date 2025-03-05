// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stft_kernel_base.h"

#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants STFTKernelBase::GetJitConstants(const STFT_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("TRANSPOSE_FRAMES", params.transpose_frames)});

    return jit;
}

void STFTKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const STFT_params&>(params);
        auto dispatchData = CalcLaunchConfig(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

CommonDispatchData STFTKernelBase::CalcLaunchConfig(const STFT_params& params) const {
    CommonDispatchData dispatchData;
    const auto inLayout = params.inputs.front().GetLayout();
    const auto& output = params.outputs.front();
    const auto outLayout = output.GetLayout();

    OPENVINO_ASSERT(output.Dimentions() == 4);
    OPENVINO_ASSERT(output.X().v == 2);

    std::vector<std::vector<Tensor::DataChannelName>> dimsByGws;

    if (params.transpose_frames) {
        dispatchData.gws = {output.Feature().v, output.Y().v, output.Batch().v};
        dimsByGws = {{Tensor::DataChannelName::FEATURE},
                     {Tensor::DataChannelName::Y},
                     {Tensor::DataChannelName::BATCH}};
    } else {
        dispatchData.gws = {output.Y().v, output.Feature().v, output.Batch().v};
        dimsByGws = {{Tensor::DataChannelName::Y},
                     {Tensor::DataChannelName::FEATURE},
                     {Tensor::DataChannelName::BATCH}};
    }
    dispatchData.lws =
        GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, inLayout, outLayout, dimsByGws);

    return dispatchData;
}

KernelsData STFTKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const auto& prim_params = static_cast<const STFT_params&>(params);

    auto dispatchData = CalcLaunchConfig(prim_params);
    KernelData k_data = KernelData::Default<STFT_params>(params);

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
                     4,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     prim_params.is_shape_agnostic);

    return {k_data};
}

bool STFTKernelBase::Validate(const Params& p) const {
    return p.GetType() == KernelType::STFT;
}

}  // namespace kernel_selector
