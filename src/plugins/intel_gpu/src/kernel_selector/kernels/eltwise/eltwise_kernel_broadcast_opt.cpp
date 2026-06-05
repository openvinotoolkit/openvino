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
    if (period % 8 != 0)                      // suffix includes X -> multiple of 8 expected
        return false;
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

    // Output element count must be a multiple of the half8 vector width.
    if (ew.outputs[0].LogicalSize() % 8 != 0) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
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
    jit.AddConstant(MakeJitConstant("BCAST_SCALAR", kind == 0 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("BCAST_SAME_SHAPE", kind == 1 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("PERIOD", static_cast<int>(period)));
    return jit;
}

EltwiseKernelBase::DispatchData EltwiseKernelBroadcastOpt::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;
    const size_t total = params.outputs[0].LogicalSize();
    const size_t n_work_items = (total + 7) / 8;  // one half8 per work-item
    dispatchData.gws = {n_work_items, 1, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

KernelsData EltwiseKernelBroadcastOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority EltwiseKernelBroadcastOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}
}  // namespace kernel_selector
