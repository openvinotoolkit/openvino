// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_memcpy.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static constexpr size_t kMemcpyVecSize = 8;
static constexpr size_t kMinXForMemcpy = 32;
static constexpr size_t kBatchRepeatInputBytesThreshold = 16 * 1024;
static constexpr size_t kWIsPerRow = 32;

ParamsKey BroadcastKernelMemcpy::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

namespace {

// Optimization decisions for the memcpy kernel. For static shapes these are
// computed from actual sizes; for dynamic shapes only the always-safe baseline
// (plain vstore8 path) is used.
struct MemcpyChoices {
    size_t batch_repeat = 1;
    bool is_x_broadcast = false;
    bool x_broadcast_block_write = false;
    size_t num_rows = 0;
    size_t out_x = 0;
};

MemcpyChoices PickStaticChoices(const broadcast_params& p) {
    const auto& input = p.inputs[0];
    const auto& output = p.outputs[0];

    MemcpyChoices c;
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
    c.x_broadcast_block_write = c.is_x_broadcast && (c.out_x % 128 == 0);
    return c;
}

CommonDispatchData ComputeDispatch(const MemcpyChoices& c) {
    CommonDispatchData dispatchData;
    if (c.x_broadcast_block_write) {
        dispatchData.gws = { 16, c.num_rows, 1 };
        dispatchData.lws = { 16, 1, 1 };
    } else {
        const size_t work_items_per_row = (c.out_x + kMemcpyVecSize - 1) / kMemcpyVecSize;
        const size_t gws_x = std::min(work_items_per_row, kWIsPerRow);
        dispatchData.gws = { gws_x, c.num_rows, 1 };
        dispatchData.lws = { gws_x, 1, 1 };
    }
    return dispatchData;
}

}  // namespace

bool BroadcastKernelMemcpy::Validate(const Params& params) const {
    const auto& p = static_cast<const broadcast_params&>(params);
    const auto& input = p.inputs[0];
    const auto& output = p.outputs[0];

    if (input.GetLayout() != output.GetLayout())
        return false;

    if (!p.fused_ops.empty())
        return false;

    // Memcpy kernel does not implement axis-permutation logic that explicit-mode broadcasts require
    // (canonicalize_shapes places the dynamic-rank input in a position the kernel doesn't reorder).
    // Numpy-mode broadcasts always have identity axis ordering and are safe.
    if (p.is_explicit_mode)
        return false;

    if (input.is_dynamic() || output.is_dynamic()) {
        // Dynamic numpy-mode: kernel handles X-broadcast at runtime via INPUT0_SIZE_X==1
        // branch, and outer dims via per-row modulo. Size-dependent optimizations
        // (BATCH_REPEAT, JIT-time IS_X_BROADCAST splat, BLOCK_WRITE) are disabled —
        // win comes from the per-row vload8/vstore8 vs ref's per-element get_idx_pos.
        return true;
    }

    if (output.X().v < kMinXForMemcpy)
        return false;

    // X-broadcast splat: input.X==1, output.X>=32.
    bool is_x_broadcast = (input.X().v == 1) && (output.X().v > 1);
    if (is_x_broadcast)
        return true;

    // Standard memcpy path requires X dimensions to match.
    if (input.X().v != output.X().v)
        return false;

    // Don't use memcpy kernel for Y-only broadcast — ref kernel's Y-blocking is faster
    bool is_y_only_broadcast = (input.Batch().v == output.Batch().v)
        && (input.Feature().v == output.Feature().v)
        && (input.Y().v != output.Y().v);
    if (is_y_only_broadcast)
        return false;

    // Only use memcpy kernel when batch-repeat gives benefit (large input, batch broadcast)
    // Otherwise ref kernel is faster due to lower per-element overhead
    const size_t input_bytes = input.LogicalSize() * BytesPerElement(input.GetDType());
    const bool has_batch_repeat = (input.Batch().v == 1) && (output.Batch().v > 1)
                                && (input_bytes > kBatchRepeatInputBytesThreshold);
    if (!has_batch_repeat)
        return false;

    return true;
}

KernelsPriority BroadcastKernelMemcpy::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}

void BroadcastKernelMemcpy::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const broadcast_params&>(params);
        // For dynamic shapes, the kernel was JIT-compiled with the safe baseline
        // (BATCH_REPEAT=1, IS_X_BROADCAST=0, X_BROADCAST_BLOCK_WRITE=0). Recompute
        // num_rows from runtime shapes; do not enable size-dependent optimizations.
        MemcpyChoices c;
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

KernelsData BroadcastKernelMemcpy::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::BROADCAST);

    const auto& prim_params = static_cast<const broadcast_params&>(params);

    if (!Validate(params))
        return {};

    MemcpyChoices choices;
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
    cldnn_jit.AddConstant(MakeJitConstant("MEMCPY_VEC_SIZE", kMemcpyVecSize));
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
