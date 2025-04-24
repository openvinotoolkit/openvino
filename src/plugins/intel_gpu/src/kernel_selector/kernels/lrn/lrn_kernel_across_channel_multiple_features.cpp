// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_kernel_across_channel_multiple_features.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey LRNKernelAcrossChannelMultipleFeatures::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableLRNMode(LRNMode::ACROSS_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey LRNKernelAcrossChannelMultipleFeatures::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_reqd_subgroup_size();

    return k;
}

static unsigned int GetOfmPerSimd(const lrn_params& params) {
    const auto& output = params.outputs[0];
    const auto local_size = params.localSize;

    if ((output.Feature().v % 8 == 0) && local_size > 4) {
        return 8;
    } else if ((output.Feature().v % 4 == 0) && local_size > 2) {
        return 4;
    } else if ((output.Feature().v % 2 == 0) && local_size > 1) {
        return 2;
    }

    return 1;
}

CommonDispatchData LRNKernelAcrossChannelMultipleFeatures::SetDefault(const lrn_params& params) const {
    CommonDispatchData dispatchData = LRNKernelBase::SetDefault(params);
    const auto& input = params.inputs[0];

    unsigned int ofm_per_simd = GetOfmPerSimd(params);

    if (input.GetLayout() == DataLayout::bfyx || input.GetLayout() == DataLayout::b_fs_yx_fsv4) {
        const auto& out = params.outputs[0];
        const unsigned int alignment = out.X().v > 16 ? 32 : 16;

        dispatchData.gws[0] = Align(out.X().v, alignment);
        dispatchData.gws[1] = out.Y().v;
        dispatchData.gws[2] = (out.Feature().v * out.Batch().v) / ofm_per_simd;

        dispatchData.lws[0] = alignment;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    } else if (input.GetLayout() == DataLayout::yxfb) {
        dispatchData.gws[0] /= ofm_per_simd;
        dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
        while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
            --dispatchData.lws[0];
        }
    }

    return dispatchData;
}

bool LRNKernelAcrossChannelMultipleFeatures::Validate(const Params& p) const {
    if (!LRNKernelBase::Validate(p)) {
        return false;
    }

    const lrn_params& params = static_cast<const lrn_params&>(p);
    if (params.localSize > 32) {
        return false;
    }

    return true;
}

JitConstants LRNKernelAcrossChannelMultipleFeatures::GetJitConstants(const lrn_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    const auto& input = params.inputs[0];
    const auto& input_dt = params.inputs[0].GetDType();
    const auto& output = params.outputs[0];

    unsigned int ofm_per_simd = GetOfmPerSimd(params);
    jit.AddConstant(MakeJitConstant("OFM_PER_SIMD", ofm_per_simd));

    if ((input.GetLayout() == DataLayout::bfyx || input.GetLayout() == DataLayout::b_fs_yx_fsv4) &&
        output.X().v <= 16) {
        jit.AddConstant(MakeJitConstant("FORCE_SIMD_16", 1));
    }

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"", {"batch_id", "feature_id + j", "y", "x"}, "lrn_result", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData LRNKernelAcrossChannelMultipleFeatures::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority LRNKernelAcrossChannelMultipleFeatures::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
