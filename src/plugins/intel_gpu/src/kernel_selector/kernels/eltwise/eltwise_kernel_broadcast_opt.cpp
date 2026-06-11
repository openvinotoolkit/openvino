// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_broadcast_opt.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

namespace {
// Outer->inner extents for planar layouts (bfyx / bfzyx). Z == 1 for bfyx.
std::vector<size_t> outer_to_inner(const DataTensor& t) {
    return {t.Batch().v, t.Feature().v, t.Z().v, t.Y().v, t.X().v};
}

// Detects the supported broadcast multiply pattern: out[i] = a[i] * b[i % period].
// kind: 0 = scalar (period 1), 1 = same-shape (period == total), 2 = suffix broadcast.
// Returns true iff the (full, broadcast) split + the suffix condition hold.
bool analyze(const eltwise_params& p, int& full_idx, int& bc_idx, size_t& period, int& kind) {
    if (p.inputs.size() != 2 || p.outputs.size() != 1)
        return false;
    const auto& out = p.outputs[0];
    const size_t total = out.LogicalSize();
    const size_t s0 = p.inputs[0].LogicalSize();
    const size_t s1 = p.inputs[1].LogicalSize();

    if (s0 == total && s1 == total) {        // same-shape multiply
        full_idx = 0; bc_idx = 1; period = total; kind = 1; return true;
    } else if (s0 == total) {
        full_idx = 0; bc_idx = 1;
    } else if (s1 == total) {
        full_idx = 1; bc_idx = 0;
    } else {
        return false;                        // neither input spans the full output
    }

    const auto& bc = p.inputs[bc_idx];
    if (bc.LogicalSize() == 1) {             // scalar broadcast
        period = 1; kind = 0; return true;
    }

    // Suffix broadcast: the broadcast operand's unit dims must be an outer prefix and its
    // non-unit dims an innermost suffix that matches the output -> b tiles every `period`.
    const auto od = outer_to_inner(out);
    const auto bd = outer_to_inner(bc);
    bool seen_non_unit = false;
    for (size_t k = 0; k < od.size(); ++k) {
        if (!seen_non_unit) {
            if (bd[k] == 1)
                continue;                    // still in the (broadcast) prefix
            seen_non_unit = true;
        }
        if (bd[k] != od[k])
            return false;                    // suffix must match the output exactly
    }
    period = bc.LogicalSize();
    kind = 2;
    return true;
}
}  // namespace

ParamsKey EltwiseKernelBroadcastOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableTensorOffset();
    k.EnableBatching();
    k.EnableEltwiseBroadcast();
    k.EnableDynamicShapesSupport();
    return k;
}

bool EltwiseKernelBroadcastOpt::Validate(const Params& p) const {
    if (!EltwiseKernelBase::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    const auto& ew = static_cast<const eltwise_params&>(p);

    // Exactly one binary multiply, no fused ops / activations / quantization / strides.
    if (ew.operations.size() != 1 || ew.operations[0].mode != EltwiseMode::MUL ||
        !ew.fused_ops.empty() || !ew.activations.empty() || ew.int8_quantization ||
        ew.layoutBased || !ew.updateInputIds.empty()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // f16, planar (bfyx/bfzyx), contiguous (no pitches).
    auto planar = [](DataLayout l) { return l == DataLayout::bfyx || l == DataLayout::bfzyx; };
    for (const auto& t : {ew.inputs[0], ew.inputs[1], ew.outputs[0]}) {
        if (t.GetDType() != Datatype::F16 || !planar(t.GetLayout()) || t.PitchesDifferFromLogicalDims()) {
            DO_NOT_USE_THIS_KERNEL(p.layerID);
        }
    }

    int full_idx, bc_idx, kind;
    size_t period;
    if (!analyze(ew, full_idx, bc_idx, period, kind)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    return true;
}

JitConstants EltwiseKernelBroadcastOpt::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = EltwiseKernelBase::GetJitConstants(params);

    int full_idx = 0, bc_idx = 1, kind = 1;
    size_t period = params.outputs[0].LogicalSize();
    analyze(params, full_idx, bc_idx, period, kind);

    jit.AddConstant(MakeJitConstant("FULL_IS_INPUT0", full_idx == 0 ? 1 : 0));
    return jit;
}

EltwiseKernelBase::DispatchData EltwiseKernelBroadcastOpt::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;
    const size_t total = params.outputs[0].LogicalSize();
    const size_t n_work_items = (total + 7) / 8;  // one half8 per work-item
    dispatchData.gws = {n_work_items, 1, 1};
    dispatchData.lws = {512, 1, 1};
    return dispatchData;
}

void EltwiseKernelBroadcastOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const eltwise_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        int full_idx, bc_idx, kind;
        size_t period;
        analyze(prim_params, full_idx, bc_idx, period, kind);
        const size_t total = prim_params.outputs[0].LogicalSize();

        if (kd.kernels[0].params.scalars.size() >= 2) {
            kd.kernels[0].params.scalars[0].v.u32 = static_cast<uint32_t>(total);
            kd.kernels[0].params.scalars[1].v.u32 = static_cast<uint32_t>(period);
        }
    };
}

KernelsData EltwiseKernelBroadcastOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);

    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
    const bool is_dynamic = newParams.is_shape_agnostic;
    kernel.params.arguments = GetArgsDesc(static_cast<uint32_t>(newParams.inputs.size()),
                                          false,
                                          false,
                                          GetFusedPrimitiveInputsCount(params),
                                          1,
                                          is_dynamic);

    int full_idx, bc_idx, kind;
    size_t period;
    analyze(newParams, full_idx, bc_idx, period, kind);
    const size_t total = newParams.outputs[0].LogicalSize();

    ScalarDescriptor total_scalar;
    total_scalar.t = ScalarDescriptor::Types::UINT32;
    total_scalar.v.u32 = static_cast<uint32_t>(total);
    kernel.params.scalars.push_back(total_scalar);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

    ScalarDescriptor period_scalar;
    period_scalar.t = ScalarDescriptor::Types::UINT32;
    period_scalar.v.u32 = static_cast<uint32_t>(period);
    kernel.params.scalars.push_back(period_scalar);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1});

    return {kd};
}

KernelsPriority EltwiseKernelBroadcastOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}
}  // namespace kernel_selector
