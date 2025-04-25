// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_mmad_b_fs_yx_fsv32_dw.h"
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>
#include <kernel_selector_utils.h>

namespace kernel_selector {

ParamsKey ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableDifferentTypes();
    k.EnableGroupedConvolution();
    k.EnableDifferentInputWeightsTypes();
    return k;
}


bool ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        return false;
    }

    auto params = dynamic_cast<const convolution_params&>(p);

    if (params.inputs[0].Feature().v != params.groups || params.outputs[0].Feature().v != params.groups)
        return false;

    if ((params.quantization == QuantizationType::ASYMMETRIC_DATA || params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS)
        && !params.HasCompensation()) {
        return false;
    }

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::SetDefault(const convolution_params& cp,
                                                                                        int /*autoTuneIndex*/) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);
    auto in_layout = cp.inputs[0].GetLayout();
    auto out_layout = cp.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::BATCH }};

    dispatchData.gws = { cp.outputs[0].Feature().v, cp.outputs[0].X().v * cp.outputs[0].Y().v, cp.outputs[0].Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, cp.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsPriority ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

// TODO: optimize this kernel
JitConstants ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::GetJitConstants(const convolution_params& params,
                                                                      const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"", {"b", "f", "y", "x"}, "res", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return jit;
}


KernelsData ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::GetKernelsData(const Params& params) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params);
    return kd;
}

}  // namespace kernel_selector
