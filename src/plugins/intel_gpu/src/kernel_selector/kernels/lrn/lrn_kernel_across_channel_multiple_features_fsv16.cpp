// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_kernel_across_channel_multiple_features_fsv16.h"
#include "kernel_selector_utils.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey LRNKernelAcrossChannelMultipleFeaturesFSV16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableLRNMode(LRNMode::ACROSS_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey LRNKernelAcrossChannelMultipleFeaturesFSV16::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_reqd_subgroup_size();

    return k;
}

CommonDispatchData LRNKernelAcrossChannelMultipleFeaturesFSV16::SetDefault(const lrn_params& params) const {
    CommonDispatchData dispatchData = LRNKernelBase::SetDefault(params);
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X },
                                                                     { Tensor::DataChannelName::Y, Tensor::DataChannelName::BATCH }};

    const auto& out = params.outputs[0];
    const unsigned int alignment = 16;

    dispatchData.gws = { Align(out.Feature().v, alignment),
                         out.X().v,
                         out.Y().v * out.Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsPriority LRNKernelAcrossChannelMultipleFeaturesFSV16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}

JitConstants LRNKernelAcrossChannelMultipleFeaturesFSV16::GetJitConstants(const lrn_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = LRNKernelBase::GetJitConstants(params, dispatchData);
    const auto& input_dt = params.inputs[0].GetDType();

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"", {"batch_id", "feature_id", "y", "x"}, "lrn_result", input_dt};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}
}  // namespace kernel_selector
