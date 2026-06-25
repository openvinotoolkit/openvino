// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_opt.h"
#include "kernel_selector_utils.h"

#include <numeric>
#include <vector>

namespace kernel_selector {

static constexpr size_t kOptVecSize = 8;
static constexpr size_t kMinXForOpt = 32;
static constexpr size_t kBatchRepeatInputBytesThreshold = 16 * 1024;
static constexpr size_t kWIsPerRow = 32;

ParamsKey BroadcastKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    // Without fused ops, OpenVINO guarantees input dtype == output dtype (else a Convert
    // is inserted by the plugin). The kernel only needs the data size, so accept any type.
    k.EnableAllInputDataType();
    k.EnableAllOutputDataType();

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

namespace {

// Optimization decisions for the opt kernel. For static shapes these are
// computed from actual sizes; for dynamic shapes only the always-safe baseline
// (plain vstore8 path) is used.
struct OptChoices {
    size_t batch_repeat = 1;
    bool is_x_broadcast = false;
    bool x_broadcast_block_write = false;
    size_t num_rows = 0;
    size_t out_x = 0;
};

OptChoices PickStaticChoices(const broadcast_params& p) {
    const auto& input = p.inputs[0];
    const auto& output = p.outputs[0];

    OptChoices c;
    c.out_x = output.X().v;
    c.is_x_broadcast = (input.X().v == 1) && (output.X().v > 1);

    // Skip batch_repeat for X-broadcast — input is already tiny per row.
    if (!c.is_x_broadcast) {
        const size_t input_bytes = input.LogicalSize() * BytesPerElement(input.GetDType());
        if (input.Batch().v == 1 && output.Batch().v > 1 && input_bytes > kBatchRepeatInputBytesThreshold) {
            c.batch_repeat = output.Batch().v;
        }
    }
    const size_t dispatch_batch = output.Batch().v / c.batch_repeat;
    c.num_rows = output.Y().v * output.Z().v * output.W().v * output.Feature().v * dispatch_batch;
    // Block-write path: handles aligned 128-element chunks plus a scalar tail.
    // Gate requires X >= 128 to make the block-write loop meaningful; smaller X
    // takes the vstore8 splat path which is fine for tiny rows.
    // intel_sub_group_block_write requires a contiguous, offset-0 output row — reject if the
    // output has padding or a buffer offset, else fall back to the vstore8 splat path.
    const bool output_contiguous = !output.PitchesDifferFromLogicalDims() && output.GetFirstElementOffset() == 0;
    c.x_broadcast_block_write = c.is_x_broadcast && (c.out_x >= 128) && output_contiguous;
    return c;
}

CommonDispatchData ComputeDispatch(const OptChoices& c) {
    CommonDispatchData dispatchData;
    if (c.x_broadcast_block_write) {
        dispatchData.gws = { 16, c.num_rows, 1 };
        dispatchData.lws = { 16, 1, 1 };
    } else {
        const size_t work_items_per_row = (c.out_x + kOptVecSize - 1) / kOptVecSize;
        const size_t gws_x = std::min(work_items_per_row, kWIsPerRow);
        dispatchData.gws = { gws_x, c.num_rows, 1 };
        dispatchData.lws = { gws_x, 1, 1 };
    }
    return dispatchData;
}

}  // namespace

bool BroadcastKernelOpt::Validate(const Params& params) const {
    const auto& p = static_cast<const broadcast_params&>(params);
    const auto& input = p.inputs[0];
    const auto& output = p.outputs[0];

    if (input.GetLayout() != output.GetLayout())
        return false;

    if (!p.fused_ops.empty())
        return false;

    // Without fused ops the kernel does pure value-broadcast (no type cast).
    // Different dtypes would require a Convert that the plugin would have already inserted.
    if (input.GetDType() != output.GetDType())
        return false;

    // The kernel maps output axis i directly to input axis i (no permutation). Two distinct
    // cases can violate that, so both guards are required:
    //  1. Numpy mode with broadcast_axes set (e.g. {2}) produces a permuted input_order like
    //     [1,2,0,3] — caught by the identity check below.
    //  2. Explicit mode (axes_mapping set) leaves broadcast_axes empty, so input_order looks
    //     like identity [0,1,2,...], but canonicalize_shapes places the input dim elsewhere —
    //     caught by the is_explicit_mode flag.
    if (p.is_explicit_mode)
        return false;

    std::vector<uint16_t> identity_order(p.input_order.size());
    std::iota(identity_order.begin(), identity_order.end(), 0);
    if (p.input_order != identity_order)
        return false;

    if (input.is_dynamic() || output.is_dynamic()) {
        // Dynamic numpy-mode: the kernel processes one output row (X) per work-group and
        // broadcasts outer dims via per-row modulo. It has no Y-blocking, so for a Y-only
        // broadcast the ref kernel is faster. Only opt-in when we can STATICALLY PROVE the
        // beneficial shape, otherwise defer to ref:
        //  - Y must be statically known and equal (not a Y-broadcast). If Y is dynamic we
        //    cannot rule out a Y-broadcast at compile time -> regression risk -> reject.
        //  - X, if statically known, must either match or be a 1->N broadcast (the only
        //    numpy-valid options); anything else is a non-row-coherent pattern -> reject.
        const bool y_known = !input.Y().is_dynamic && !output.Y().is_dynamic;
        if (!y_known || input.Y().v != output.Y().v)
            return false;

        const bool x_known = !input.X().is_dynamic && !output.X().is_dynamic;
        if (x_known && input.X().v != output.X().v && input.X().v != 1)
            return false;

        return true;
    }

    if (output.X().v < kMinXForOpt)
        return false;

    // X-broadcast splat: input.X==1, output.X>=32.
    bool is_x_broadcast = (input.X().v == 1) && (output.X().v > 1);
    if (is_x_broadcast)
        return true;

    // Standard vectorized path requires X dimensions to match.
    if (input.X().v != output.X().v)
        return false;

    // Don't use opt kernel for Y-only broadcast — ref kernel's Y-blocking is faster
    bool is_y_only_broadcast = (input.Batch().v == output.Batch().v)
        && (input.Feature().v == output.Feature().v)
        && (input.Y().v != output.Y().v);
    if (is_y_only_broadcast)
        return false;

    // Only use opt kernel when batch-repeat gives benefit (large input, batch broadcast)
    // Otherwise ref kernel is faster due to lower per-element overhead
    const size_t input_bytes = input.LogicalSize() * BytesPerElement(input.GetDType());
    const bool has_batch_repeat = (input.Batch().v == 1) && (output.Batch().v > 1)
                                && (input_bytes > kBatchRepeatInputBytesThreshold);
    if (!has_batch_repeat)
        return false;

    return true;
}

KernelsPriority BroadcastKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}

void BroadcastKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const broadcast_params&>(params);
        // For dynamic shapes, the kernel was JIT-compiled with the safe baseline
        // (BATCH_REPEAT=1, IS_X_BROADCAST=0, X_BROADCAST_BLOCK_WRITE=0). Recompute
        // num_rows from runtime shapes; do not enable size-dependent optimizations.
        OptChoices c;
        const auto& output = prim_params.outputs[0];
        c.out_x = output.X().v;
        c.num_rows = output.Y().v * output.Z().v * output.W().v
                   * output.Feature().v * output.Batch().v;
        const auto dispatchData = ComputeDispatch(c);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData BroadcastKernelOpt::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::BROADCAST);

    const auto& prim_params = static_cast<const broadcast_params&>(params);

    if (!Validate(params))
        return {};

    OptChoices choices;
    if (!prim_params.has_dynamic_tensors()) {
        choices = PickStaticChoices(prim_params);
    } else {
        // Safe baseline for dynamic shapes; final num_rows recomputed at dispatch time.
        choices.batch_repeat = 1;
        choices.is_x_broadcast = false;
        choices.x_broadcast_block_write = false;
        choices.num_rows = 1;  // placeholder; update_dispatch_data overwrites
        choices.out_x = 1;
    }

    const auto dispatchData = ComputeDispatch(choices);

    KernelData k_data = KernelData::Default<broadcast_params>(params);
    GetUpdateDispatchDataFunc(k_data);

    auto cldnn_jit = MakeBaseParamsJitConstants(prim_params);
    cldnn_jit.AddConstant(MakeJitConstant("OPT_VEC_SIZE", kOptVecSize));
    cldnn_jit.AddConstant(MakeJitConstant("BATCH_REPEAT", choices.batch_repeat));
    cldnn_jit.AddConstant(MakeJitConstant("IS_X_BROADCAST", choices.is_x_broadcast ? 1 : 0));
    cldnn_jit.AddConstant(MakeJitConstant("X_BROADCAST_BLOCK_WRITE", choices.x_broadcast_block_write ? 1 : 0));
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     0,
                     1,
                     prim_params.is_shape_agnostic);

    return {k_data};
}
}  // namespace kernel_selector
