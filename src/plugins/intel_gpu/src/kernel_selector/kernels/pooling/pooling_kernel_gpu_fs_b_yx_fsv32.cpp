// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_gpu_fs_b_yx_fsv32.h"

namespace kernel_selector {
ParamsKey PoolingKerneGPU_fs_b_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
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

DeviceFeaturesKey PoolingKerneGPU_fs_b_yx_fsv32::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

PoolingKernelBase::DispatchData PoolingKerneGPU_fs_b_yx_fsv32::SetDefault(const pooling_params& params) const {
    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    dispatchData.gws[0] = params.outputs[0].X().v;  // X output blocks
    dispatchData.gws[1] = params.outputs[0].Y().v;  // Y output clocks
    // in fs_b_yx_fsv32 format we will process 2 features per work item, so reads/writes are done in full writes for
    // fp16
    dispatchData.gws[2] = RoundUp(params.outputs[0].Feature().v, 32) * params.outputs[0].Batch().v / 2;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 16;

    return dispatchData;
}

bool PoolingKerneGPU_fs_b_yx_fsv32::Validate(const Params& p) const {
    if (!PoolingKernelBase::Validate(p))
        return false;

    auto pp = static_cast<const pooling_params&>(p);

    // Feature padding before must be aligned to 32 to keep slices aligned
    if (pp.outputs[0].Feature().pad.before % 32 != 0)
        return false;

    if (pp.inputs[0].Feature().pad.before % 32 != 0)
        return false;

    return true;
}

JitConstants PoolingKerneGPU_fs_b_yx_fsv32::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);
    auto pp = static_cast<const pooling_params&>(params);

    // Heurestic needed for very big pool size.
    // ToDo Can it be changed to lower pool sizes?
    if (pp.poolSize.x >= 7 && pp.poolSize.y >= 7 && pp.poolType == PoolType::AVG) {
        jit.AddConstant(MakeJitConstant("USE_FLOAT_ACC", true));
    }
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = {"",
                                     {"b", "((fs * 32) + sglid)", "out_y", "out_x"},
                                     "pool_result",
                                     input_dt,
                                     2,
                                     LoadType::LT_ALIGNED_READ,
                                     BoundaryCheck::ENABLED,
                                     IndexType::TENSOR_COORD,
                                     Tensor::DataChannelName::FEATURE};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData PoolingKerneGPU_fs_b_yx_fsv32::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority PoolingKerneGPU_fs_b_yx_fsv32::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
