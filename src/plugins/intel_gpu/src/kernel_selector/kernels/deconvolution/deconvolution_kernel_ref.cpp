// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_kernel_ref.h"

namespace kernel_selector {

ParamsKey DeconvolutionKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    return k;
}

CommonDispatchData DeconvolutionKernelRef::SetDefault(const deconvolution_params& params) const {
    CommonDispatchData dispatchData = DeconvolutionKernelBase::SetDefault(params);

    if (params.outputs[0].Feature().v * params.outputs[0].Batch().v <= 16) {
        const auto& out = params.outputs[0];
        dispatchData.gws[0] = Align(out.X().v, 32);
        dispatchData.gws[1] = out.Y().v * out.Z().v;
        dispatchData.gws[2] = out.Feature().v * out.Batch().v;

        dispatchData.lws[0] = 32;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}

KernelsPriority DeconvolutionKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants DeconvolutionKernelRef::GetJitConstants(const deconvolution_params& params) const {
    auto jit = DeconvolutionKernelBase::GetJitConstants(params);

    if (params.outputs[0].Feature().v * params.outputs[0].Batch().v <= 16)
        jit.AddConstant(MakeJitConstant("DIM_ORDER_XYBF", 1));

    if (!params.fused_ops.empty()) {
        auto fused_dt = GetActivationType(params);
        std::vector<std::string> idx_order;
        if (params.outputs[0].Dimentions() <= 4) {
            idx_order = { "batch_offset", "ofm_offset", "out_y", "out_x" };
        } else {
            idx_order = { "batch_offset", "ofm_offset", "out_z", "out_y", "out_x" };
        }
        FusedOpsConfiguration conf = { "", idx_order, "post_activation", fused_dt, 1, LoadType::LT_UNALIGNED, BoundaryCheck::DISABLED };

        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}
}  // namespace kernel_selector
