// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_bfyx_to_blocked_format.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <functional>
#include <cmath>

// Tile size : 4x4 or 8x8
#define MIN_TILE_SIZE 4
#define DEFAULT_TILE_SIZE 8

namespace kernel_selector {
ParamsKey ReorderKernel_bfyx_to_blocked_format::GetSupportedKey() const {
    ParamsKey k;

    k.EnableAllInputDataType();
    k.EnableAllOutputDataType();

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);

    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);

    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorOffset();
    k.EnableTensorPitches();

    return k;
}

static inline std::string GetTiledInputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
    case 4:
        order_str = "b, f + lh, y, x";
        break;
    case 5:
        order_str = "b, f + lh, z, y, x";
        break;
    default: throw std::runtime_error("Unsupported combination\n");
    }
    return order_str;
}

static inline size_t GetFsvAlignment(const reorder_params& params) {
    const auto& out = params.outputs[0];
    int fsv_alignment = -1;
    switch (out.GetLayout()) {
    case DataLayout::b_fs_yx_fsv4:
        fsv_alignment = 4;
        break;
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv32_fsv16:
        fsv_alignment = 16;
        break;
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::fs_b_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
    case DataLayout::bs_fs_yx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv32_fsv32:
        fsv_alignment = 32;
        break;
    default:
        throw std::runtime_error("Unsupported combination\n");
    }
    return fsv_alignment;
}

static inline size_t GetBsvAlignment(const reorder_params& params) {
    const auto& out = params.outputs[0];
    int alignment = -1;
    switch (out.GetLayout()) {
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_yx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv16_fsv32:
        alignment = 16;
        break;
    case DataLayout::bs_fs_zyx_bsv32_fsv32:
    case DataLayout::bs_fs_zyx_bsv32_fsv16:
        alignment = 32;
        break;
    default:
        throw std::runtime_error("Unsupported combination\n");
    }
    return alignment;
}

static inline size_t GetTileSize(const reorder_params& params) {
    const Datatype input_type = params.inputs[0].GetDType();
    const Datatype output_type = params.outputs[0].GetDType();

    // i64 supports tile size 4
    if ((input_type == Datatype::INT64) || (output_type == Datatype::INT64)) {
        return MIN_TILE_SIZE;
    }

    if (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv4) {
        return MIN_TILE_SIZE;
    }

    if (params.inputs[0].Feature().v < DEFAULT_TILE_SIZE) {
        return MIN_TILE_SIZE;
    }

    return DEFAULT_TILE_SIZE;
}

static inline std::vector<size_t> GetGWS(const reorder_params& params) {
    const auto& in = params.inputs[0];
    const size_t tile_size = GetTileSize(params);
    const size_t fsv_alignment = GetFsvAlignment(params);

    std::vector<size_t> gws = { (fsv_alignment / tile_size),
        CeilDiv(in.X().v, tile_size) * in.Y().v * in.Z().v,
        in.Batch().v * CeilDiv(in.Feature().v, fsv_alignment) };

    return gws;
}

static std::vector<size_t> GetBestLwsFromGws(const reorder_params& params, const std::vector<size_t>& gws, const size_t tile_width, const size_t tile_size) {
    std::vector<size_t> lws{ 1, 1, 1 };
    std::vector<size_t> dims{ 0, 1, 2 };

    // SLM size: elemsize * tile_width * tile_width * work_items <= 64K
    const size_t elem_size = params.outputs[0].ElementSize();
    const size_t max_local_mem_size = params.engineInfo.maxLocalMemSize;
    const size_t max_work_group_size = params.engineInfo.maxWorkGroupSize;
    size_t max_num_work_items = std::min(max_work_group_size, max_local_mem_size / (elem_size * tile_width * tile_size));

    for (size_t i = 0; i < dims.size(); ++i) {
        size_t dim = dims[i];
        size_t max_divider = static_cast<size_t>(std::sqrt(gws[dim]) + 1);
        for (size_t divider = 1; divider <= max_divider; ++divider) {
            if (gws[dim] % divider == 0) {
                const size_t lws0 = gws[dim] / divider;
                if (lws0 <= max_num_work_items) {
                    lws[dim] = std::max(lws[dim], lws0);
                }
                if (divider <= max_num_work_items) {
                    lws[dim] = std::max(lws[dim], divider);
                }
            }
        }
        max_num_work_items /= lws[dim];
    }
    return lws;
}

CommonDispatchData ReorderKernel_bfyx_to_blocked_format::SetDefault(const reorder_params& params) const {
    CommonDispatchData dispatchData;
    const size_t tile_size = GetTileSize(params);
    dispatchData.gws = GetGWS(params);
    dispatchData.lws = GetBestLwsFromGws(params, dispatchData.gws, tile_size, tile_size);
    return dispatchData;
}

JitConstants ReorderKernel_bfyx_to_blocked_format::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);

    const size_t b = params.inputs[0].Batch().v;
    const size_t f = params.inputs[0].Feature().v;
    const size_t x = params.inputs[0].X().v;
    const size_t tile_size = GetTileSize(params);
    const size_t input_ndims = params.inputs[0].GetDims().size();
    const size_t fsv_alignment = GetFsvAlignment(params);

    const auto& gws = GetGWS(params);
    const auto& lws = GetBestLwsFromGws(params, gws, tile_size, tile_size);
    const uint64_t total_lws = lws[0] * lws[1] * lws[2];

    jit.AddConstant(MakeJitConstant("INPUT0_TILED_ORDER", GetTiledInputOrder(input_ndims)));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_SLICE_NUM", CeilDiv(f, fsv_alignment)));
    jit.AddConstant(MakeJitConstant("TILE_SIZE", tile_size));
    jit.AddConstant(MakeJitConstant("FSV_ALIGNMENT", fsv_alignment));
    jit.AddConstant(MakeJitConstant("TRANS_BUF_SIZE", tile_size * total_lws));

    if (params.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        jit.AddConstant(MakeJitConstant("FS_B_YX_FSV", 1));
    }

    if (params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
        const size_t bsv_alignment = GetBsvAlignment(params);
        jit.AddConstant(MakeJitConstant("DOUBLE_BLOCKED_FORMAT", 1));
        jit.AddConstant(MakeJitConstant("INPUT0_BATCH_SLICE_NUM", CeilDiv(b, bsv_alignment)));
        jit.AddConstant(MakeJitConstant("BSV_ALIGNMENT", bsv_alignment));
    }

    // whether F is tile_size-aligned
    if (f % tile_size == 0) {
        jit.AddConstant(MakeJitConstant("F_NO_REMAINDER_CONDITION", "(f < INPUT0_FEATURE_NUM)"));
    } else {
        jit.AddConstant(MakeJitConstant("F_REMAINDER_SIZE", f % tile_size));
        jit.AddConstant(MakeJitConstant("F_REMAINDER_CONDITION", "(f >= (INPUT0_FEATURE_NUM - F_REMAINDER_SIZE)) && (f < INPUT0_FEATURE_NUM)"));
        jit.AddConstant(MakeJitConstant("F_NO_REMAINDER_CONDITION", "(f < (INPUT0_FEATURE_NUM - F_REMAINDER_SIZE))"));
    }

    // whether x is tile_size-aligned
    if (x % tile_size != 0) {
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE", x % tile_size));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_CONDITION", "(x >= (INPUT0_SIZE_X - X_REMAINDER_SIZE)) && (x < INPUT0_SIZE_X)"));
        jit.AddConstant(MakeJitConstant("X_NO_REMAINDER_CONDITION", "(x < (INPUT0_SIZE_X - X_REMAINDER_SIZE))"));
    }

    return jit;
}

KernelsData ReorderKernel_bfyx_to_blocked_format::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    return GetCommonKernelsData(orgParams);
}

bool ReorderKernel_bfyx_to_blocked_format::Validate(const Params& p) const {
    if (!ReorderKernelBase::Validate(p)) {
        return false;
    }

    const reorder_params& params = static_cast<const reorder_params&>(p);
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (input.GetDims().size() != output.GetDims().size()) {
        return false;
    }

    // padding is not supported
    if (input.X().pad.before != 0 || input.X().pad.after != 0 ||
        input.Y().pad.before != 0 || input.Y().pad.after != 0 ||
        input.Z().pad.before != 0 || input.Z().pad.after != 0 ||
        input.W().pad.before != 0 || input.W().pad.after != 0 ||
        input.Feature().pad.before != 0 || input.Feature().pad.after != 0 ||
        input.Batch().pad.before != 0 || input.Batch().pad.after != 0) {
        return false;
    }

    if (output.X().pad.before != 0 || output.X().pad.after != 0 ||
        output.Y().pad.before != 0 || output.Y().pad.after != 0 ||
        output.Z().pad.before != 0 || output.Z().pad.after != 0 ||
        output.W().pad.before != 0 || output.W().pad.after != 0 ||
        output.Feature().pad.before != 0 || output.Feature().pad.after != 0 ||
        output.Batch().pad.before != 0 || output.Batch().pad.after != 0) {
        return false;
    }

    return true;
}

KernelsPriority ReorderKernel_bfyx_to_blocked_format::GetKernelsPriority(const Params& p) const {
    const reorder_params& params = static_cast<const reorder_params&>(p);
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    const size_t b = input.Batch().v;
    const size_t f = input.Feature().v;
    const size_t x = input.X().v;
    const size_t y = input.Y().v;
    const size_t z = input.Z().v;

    const size_t elem_size = input.ElementSize();
    const size_t total_data_byte = b * f * x * y * z * elem_size;

    const size_t tile_size = GetTileSize(params);
    const size_t fsv_alignment = GetFsvAlignment(params);

    if ((f < fsv_alignment && x < tile_size) || total_data_byte < 32000) {
        return DONT_USE_IF_HAVE_SOMETHING_ELSE;
    }

    // At this condition, reorder_data_fast_b1 is faster
    if (b == 1 && output.Batch().v == 1 && params.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16 && f < 256) {
        return FORCE_PRIORITY_8;
    }

    return FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
