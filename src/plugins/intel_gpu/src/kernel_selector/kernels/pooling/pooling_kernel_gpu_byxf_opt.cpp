// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_gpu_byxf_opt.h"

namespace kernel_selector {
ParamsKey PoolingKernelGPUByxfOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    return k;
}

PoolingKernelBase::DispatchData PoolingKernelGPUByxfOpt::SetDefault(const pooling_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    dispatchData.gws[2] = output.Batch().v * (CeilDiv(output.Feature().v, 8));

    return dispatchData;
}

JitConstants PoolingKernelGPUByxfOpt::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = {"",
                                     {"b", "f + i", "y", "x"},
                                     "pool_result",
                                     input_dt,
                                     1,
                                     LoadType::LT_UNALIGNED,
                                     BoundaryCheck::ENABLED,
                                     IndexType::TENSOR_COORD,
                                     Tensor::DataChannelName::FEATURE};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

bool PoolingKernelGPUByxfOpt::Validate(const Params& p) const {
    if (!PoolingKernelBase::Validate(p)) {
        return false;
    }

    const pooling_params& params = static_cast<const pooling_params&>(p);
    if (params.inputs[0].Feature().v % 8 != 0) {
        return false;
    }

    if (NeedsBoundaryCheck(params)) {
        return false;
    }

    return true;
}

KernelsData PoolingKernelGPUByxfOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority PoolingKernelGPUByxfOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
