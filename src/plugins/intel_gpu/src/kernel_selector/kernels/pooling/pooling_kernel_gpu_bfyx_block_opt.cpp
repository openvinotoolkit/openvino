// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_gpu_bfyx_block_opt.h"

namespace kernel_selector {
ParamsKey PoolingKernelGPUBfyxBlockOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableDifferentTypes();
    return k;
}

PoolingKernelBase::DispatchData PoolingKernelGPUBfyxBlockOpt::SetDefault(const pooling_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    dispatchData.gws[1] = CeilDiv(output.Y().v, params.poolSize.y);

    return dispatchData;
}

JitConstants PoolingKernelGPUBfyxBlockOpt::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);

    jit.AddConstant(
        MakeJitConstant("BLOCK_SIZE_Y", params.poolSize.y + params.poolSize.y * params.poolStride.y - 1));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = {"",
                                     {"b", "f", "y + i", "x"},
                                     "pool_result",
                                     input_dt,
                                     1,
                                     LoadType::LT_UNALIGNED,
                                     BoundaryCheck::ENABLED,
                                     IndexType::TENSOR_COORD,
                                     Tensor::DataChannelName::Y};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

bool PoolingKernelGPUBfyxBlockOpt::Validate(const Params& p) const {
    if (!PoolingKernelBase::Validate(p)) {
        return false;
    }

    const pooling_params& params = static_cast<const pooling_params&>(p);
    if (NeedsBoundaryCheck(params) || params.poolSize.x > 5 || params.poolSize.y > 5 || params.poolSize.x < 3 ||
        params.poolSize.y < 3) {
        return false;
    }

    return true;
}

KernelsData PoolingKernelGPUBfyxBlockOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority PoolingKernelGPUBfyxBlockOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
