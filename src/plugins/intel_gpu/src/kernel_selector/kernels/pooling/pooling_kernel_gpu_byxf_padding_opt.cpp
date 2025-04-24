// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_gpu_byxf_padding_opt.h"

namespace kernel_selector {
ParamsKey PoolingKernelGPUByxfPaddingOpt::GetSupportedKey() const {
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
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableDifferentTypes();
    return k;
}

PoolingKernelBase::DispatchData PoolingKernelGPUByxfPaddingOpt::SetDefault(const pooling_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    dispatchData.gws[2] = output.Batch().v * (CeilDiv(output.Feature().v, 8));

    return dispatchData;
}

JitConstants PoolingKernelGPUByxfPaddingOpt::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = {"", {"b", "f + i", "y", "x"}, "pool_result", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

bool PoolingKernelGPUByxfPaddingOpt::Validate(const Params& p) const {
    if (!PoolingKernelBase::Validate(p)) {
        return false;
    }

    const pooling_params& params = static_cast<const pooling_params&>(p);
    if (params.inputs[0].Feature().v % 8 != 0)
        return false;

    return true;
}

KernelsData PoolingKernelGPUByxfPaddingOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority PoolingKernelGPUByxfPaddingOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
