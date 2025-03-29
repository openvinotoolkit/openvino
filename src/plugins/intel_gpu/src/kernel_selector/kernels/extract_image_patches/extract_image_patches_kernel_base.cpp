// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ExtractImagePatchesKernelBase::GetSupportedKey() const {
    ParamsKey k;

    k.EnableAllInputDataType();
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ExtractImagePatchesKernelBase::GetJitConstants(const extract_image_patches_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("SIZE_ROWS", params.sizes[0]),
        MakeJitConstant("SIZE_COLS", params.sizes[1]),
        MakeJitConstant("STRIDE_ROWS", params.strides[0]),
        MakeJitConstant("STRIDE_COLS", params.strides[1]),
        MakeJitConstant("RATES_ROWS", params.rates[0]),
        MakeJitConstant("RATES_COLS", params.rates[1]),
    });
    if (params.auto_pad == "same_upper")
        jit.AddConstant(MakeJitConstant("AUTO_PAD", 1));
    else if (params.auto_pad == "same_lower")
        jit.AddConstant(MakeJitConstant("AUTO_PAD", 2));

    return jit;
}

ExtractImagePatchesKernelBase::DispatchData ExtractImagePatchesKernelBase::SetDefault(const extract_image_patches_params& params) const {
    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y }};

    dispatchData.gws = { params.outputs[0].Batch().v,
                         params.outputs[0].Feature().v,
                         params.outputs[0].Y().v * params.outputs[0].X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsData ExtractImagePatchesKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const extract_image_patches_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData kd = KernelData::Default<extract_image_patches_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {kd};
}

bool ExtractImagePatchesKernelBase::Validate(const Params& p) const {
    const extract_image_patches_params& params = static_cast<const extract_image_patches_params&>(p);

    if (params.GetType() != KernelType::EXTRACT_IMAGE_PATCHES) {
        return false;
    }

    return true;
}
}  // namespace kernel_selector
