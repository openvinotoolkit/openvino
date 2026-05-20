// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_memcpy.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static constexpr size_t kMemcpyVecSize = 8;
static constexpr size_t kMinXForMemcpy = 32;

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

    return k;
}

bool BroadcastKernelMemcpy::Validate(const Params& params) const {
    const auto& p = static_cast<const broadcast_params&>(params);
    const auto& input = p.inputs[0];
    const auto& output = p.outputs[0];

    if (input.is_dynamic() || output.is_dynamic())
        return false;

    if (input.GetLayout() != output.GetLayout())
        return false;

    if (input.X().v != output.X().v)
        return false;

    if (output.X().v < kMinXForMemcpy)
        return false;

    // Don't use memcpy kernel for Y-only broadcast — ref kernel's Y-blocking is faster
    bool is_y_only_broadcast = (input.Batch().v == output.Batch().v)
        && (input.Feature().v == output.Feature().v)
        && (input.Y().v != output.Y().v);
    if (is_y_only_broadcast)
        return false;

    if (!p.fused_ops.empty())
        return false;

    // Only use memcpy kernel when batch-repeat gives benefit (large input, batch broadcast)
    // Otherwise ref kernel is faster due to lower per-element overhead
    size_t input_bytes = input.LogicalSize() * BytesPerElement(input.GetDType());
    bool has_batch_repeat = (input.Batch().v == 1) && (output.Batch().v > 1) && (input_bytes > 16 * 1024);
    if (!has_batch_repeat)
        return false;

    return true;
}

KernelsPriority BroadcastKernelMemcpy::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}

KernelsData BroadcastKernelMemcpy::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::BROADCAST);

    const auto& prim_params = static_cast<const broadcast_params&>(params);

    if (!Validate(params))
        return {};

    const auto& input = prim_params.inputs[0];
    const auto& output = prim_params.outputs[0];
    const size_t out_x = output.X().v;

    // Batch repeat: dispatch over input batch only, kernel writes to all output batches.
    // Only beneficial when input is large enough that redundant reads aren't cached.
    size_t batch_repeat = 1;
    size_t input_bytes = input.LogicalSize() * BytesPerElement(input.GetDType());
    if (input.Batch().v == 1 && output.Batch().v > 1 && input_bytes > 16 * 1024) {
        batch_repeat = output.Batch().v;
    }
    const size_t dispatch_batch = output.Batch().v / batch_repeat;

    const size_t num_rows = output.Y().v * output.Z().v * output.W().v
                          * output.Feature().v * dispatch_batch;
    const size_t work_items_per_row = (out_x + kMemcpyVecSize - 1) / kMemcpyVecSize;

    DispatchData dispatchData;
    constexpr size_t kWIsPerRow = 32;
    size_t gws_x = std::min(work_items_per_row, kWIsPerRow);
    dispatchData.gws = { gws_x, num_rows, 1 };
    dispatchData.lws = { gws_x, 1, 1 };

    KernelData k_data = KernelData::Default<broadcast_params>(params);

    auto cldnn_jit = MakeBaseParamsJitConstants(prim_params);
    cldnn_jit.AddConstant(MakeJitConstant("MEMCPY_VEC_SIZE", kMemcpyVecSize));
    cldnn_jit.AddConstant(MakeJitConstant("BATCH_REPEAT", batch_repeat));
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
                     false);

    return {k_data};
}
}  // namespace kernel_selector
