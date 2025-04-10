// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_gpu_bs_fs_yx_bsv16_fsv16.h"
#include "kernel_selector_utils.h"

//
// Kernel specific constants
//
#define SIMD_SIZE 16

namespace kernel_selector {
ParamsKey Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC_WITH_PADDING);
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_reqd_subgroup_size();

    return k;
}

PoolingKernelBase::DispatchData Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::SetDefault(const pooling_params& params) const {
    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    dispatchData.gws[0] = params.outputs[0].Feature().v/16;
    dispatchData.gws[1] = params.outputs[0].X().v * params.outputs[0].Y().v;
    dispatchData.gws[2] = params.outputs[0].Batch().v;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = SIMD_SIZE;

    return dispatchData;
}

KernelsPriority Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

JitConstants Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);

    if (!params.fused_ops.empty()) {
        auto input_dt = EnableRound(params) ? Datatype::INT32 : GetActivationType(params);
        FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "pool_result[i]", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

bool Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::Validate(const Params& params) const {
    if (!PoolingKernelBase::Validate(params)) {
        return false;
    }
    auto p = dynamic_cast<const pooling_params&>(params);

    if (p.inputs[0].Feature().v % 16 != 0)
        return false;

    if (p.inputs[0].Batch().v % 16 != 0)
        return false;

    return true;
}
}  // namespace kernel_selector
