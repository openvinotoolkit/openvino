// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_kernel_weighted_x16.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey ReduceKernelWeightedX16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableEltwiseBroadcast();
    return k;
}

bool ReduceKernelWeightedX16::Validate(const Params& p) const {
    if (!ReduceKernelBase::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& params = static_cast<const reduce_params&>(p);
    if (!params.weighted || params.is_shape_agnostic || params.inputs.size() != 2 || params.outputs.size() != 1 || params.reduceMode != ReduceMode::SUM ||
        params.reduceAxes.size() != 1 || params.reduceAxes[0] != 2 || !params.fused_ops.empty()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& values = params.inputs[0];
    const auto& weights = params.inputs[1];
    const auto& output = params.outputs[0];
    if (values.GetLayout() != DataLayout::bfyx || weights.GetLayout() != DataLayout::bfyx || output.GetLayout() != DataLayout::bfyx ||
        values.PitchesDifferFromLogicalDims() || weights.PitchesDifferFromLogicalDims() || output.PitchesDifferFromLogicalDims() ||
        values.GetFirstElementOffset() != 0 || weights.GetFirstElementOffset() != 0 || output.GetFirstElementOffset() != 0) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (values.X().v != 16 || weights.X().v != 16 || output.X().v != 1 || values.Y().v <= 1024 || values.Batch().v != weights.Batch().v ||
        values.Batch().v != output.Batch().v || weights.Feature().v != 1 || values.Feature().v != output.Feature().v || values.Y().v != weights.Y().v ||
        values.Y().v != output.Y().v) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

CommonDispatchData ReduceKernelWeightedX16::SetDefault(const reduce_params& params) const {
    CommonDispatchData dispatch;
    dispatch.gws = {params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Y().v, 1, 1};
    dispatch.lws = GetOptimalLocalWorkGroupSizes(dispatch.gws, params.engineInfo, params.inputs[0].GetLayout(), params.outputs[0].GetLayout());
    return dispatch;
}

JitConstants ReduceKernelWeightedX16::GetJitConstants(const reduce_params& params) const {
    auto jit = ReduceKernelBase::GetJitConstants(params);
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetFinalAccumulatorType(params), "FINAL_ACCUMULATOR"));
    return jit;
}

KernelsData ReduceKernelWeightedX16::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority ReduceKernelWeightedX16::GetKernelsPriority(const Params&) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
