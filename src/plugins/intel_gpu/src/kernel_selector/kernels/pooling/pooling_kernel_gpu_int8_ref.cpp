// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_gpu_int8_ref.h"
#include <string>

namespace kernel_selector {
ParamsKey PoolingKernelGPUInt8Ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
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

KernelsData PoolingKernelGPUInt8Ref::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority PoolingKernelGPUInt8Ref::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

JitConstants PoolingKernelGPUInt8Ref::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    JitConstants jit = PoolingKernelBase::GetJitConstants(params, dispatchData);
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);

        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            idx_order = {"b", "f", "y", "x"};
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order = {"b", "f", "z", "y", "x"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "pool_result", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

bool PoolingKernelGPUInt8Ref::Validate(const Params& params) const {
    if (!PoolingKernelBase::Validate(params)) {
        return false;
    }
    auto p = dynamic_cast<const pooling_params&>(params);

    if (p.inputs[0].GetDType() == Datatype::INT8 || p.inputs[0].GetDType() == Datatype::UINT8) {
        // Max pooling doesn't change quantization ranges, so output data type should be the same as input
        if (p.poolType == PoolType::MAX && p.outputs[0].GetDType() != p.inputs[0].GetDType() &&
            p.quantization == QuantizationType::NONE)
            return false;
//         Average pooling should produce FP by default. (u)int8 is possible when quantize op is fused.
//        if (p.poolType == PoolType::AVG &&
//            !((p.outputs[0].GetDType() == p.inputs[0].GetDType() && !p.fused_ops.empty()) ||
//              (p.outputs[0].GetDType() == Datatype::F32 || p.outputs[0].GetDType() == Datatype::F16)))
//            return false;
    }

    return true;
}

}  // namespace kernel_selector
