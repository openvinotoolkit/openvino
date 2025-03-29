// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_kernel_imad_ref.hpp"

#include "kernel_selector_utils.h"

#include <vector>
#include <string>

namespace kernel_selector {

ParamsKey DeconvolutionKernel_imad_ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableAllOutputLayout();

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableGroupedConvolution();

    return k;
}

WeightsLayout DeconvolutionKernel_imad_ref::GetPreferredWeightsLayout(const deconvolution_params&) const {
    return WeightsLayout::g_os_zyx_is_osv32_isv4;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernel_imad_ref::SetDefault(const deconvolution_params& params) const {
    DispatchData dispatchData = Parent::SetDefault(params);
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y, Tensor::DataChannelName::Z },
                                                                     { Tensor::DataChannelName::BATCH }};

    dispatchData.gws = {
         params.outputs[0].Feature().v,
         params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v,
         params.outputs[0].Batch().v
    };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsPriority DeconvolutionKernel_imad_ref::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

JitConstants DeconvolutionKernel_imad_ref::GetJitConstants(const deconvolution_params& params) const {
    auto jit = Parent::GetJitConstants(params);
    auto tile_ifm = GetTileIFM(params);

    jit.AddConstant(MakeJitConstant("TILE_IFM", tile_ifm));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.outputs[0].Dimentions() <= 4) {
            idx_order = { "out_b", "out_f", "out_y", "out_x" };
        } else {
            idx_order = { "out_b", "out_f", "out_z", "out_y", "out_x" };
        }
        auto conf = FusedOpsConfiguration{ "", idx_order, "dequantized", GetActivationType(params), 1, LoadType::LT_UNALIGNED, BoundaryCheck::DISABLED };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

size_t DeconvolutionKernel_imad_ref::GetTileIFM(const deconvolution_params&) const {
    return 4;
}


}  // namespace kernel_selector
