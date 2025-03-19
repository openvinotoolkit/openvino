// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_b_fs_yx_fsv16_fsv32_to_bfyx.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <functional>
#include <cmath>

// Tile size : 8x8
#define HALF_TILE_SIZE 4
#define DEFAULT_TILE_SIZE 8

namespace kernel_selector {
ParamsKey ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::INT32);

    k.EnableAllOutputDataType();

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableBatching();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();

    return k;
}

DeviceFeaturesKey ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_blocked_read_write();
    k.requires_reqd_subgroup_size();

    return k;
}

static inline std::string GetTiledOutputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
    case 4:
        order_str = "b, f, y, x";
        break;
    case 5:
        order_str = "b, f, z, y, x";
        break;
    case 6:
        order_str = "b, f, w, z, y, x";
        break;
    default: throw std::runtime_error("Unsupported size\n");
    }
    return order_str;
}

static inline size_t GetFsvAlignment(const reorder_params& params) {
    const auto& in = params.inputs[0];
    int fsv_alignment = -1;
    switch (in.GetLayout()) {
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
        fsv_alignment = 16;
        break;
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
        fsv_alignment = 32;
        break;
    default:
        throw std::runtime_error("Unsupported input\n");
    }
    return fsv_alignment;
}

static inline size_t GetTileSize(const reorder_params& params) {
    size_t tile_size = 0;

    const auto& in = params.inputs[0];
    switch (in.GetLayout()) {
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
        tile_size = DEFAULT_TILE_SIZE;
        break;
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
        tile_size = HALF_TILE_SIZE;
        break;
    default:
        throw std::runtime_error("Unsupported input\n");
    }

    return tile_size;
}

static inline std::vector<size_t> GetGWS(const reorder_params& params) {
    const auto& in = params.inputs[0];
    const size_t fsv_alignment = GetFsvAlignment(params);

    std::vector<size_t> gws = { CeilDiv(in.X().v, DEFAULT_TILE_SIZE) * fsv_alignment,
        in.Y().v * in.Z().v,
        in.Batch().v * CeilDiv(in.Feature().v, fsv_alignment) };

    return gws;
}

CommonDispatchData ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx::SetDefault(const reorder_params& params) const {
    CommonDispatchData dispatchData;
    const size_t fsv_alignment = GetFsvAlignment(params);

    dispatchData.gws = GetGWS(params);
    dispatchData.lws = { fsv_alignment, 1, 1};
    return dispatchData;
}

JitConstants ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);

    const size_t f = params.inputs[0].Feature().v;
    const size_t x = params.inputs[0].X().v;
    const size_t tile_size = GetTileSize(params);
    const size_t output_ndims = params.outputs[0].GetDims().size();
    const size_t fsv_alignment = GetFsvAlignment(params);

    jit.AddConstant(MakeJitConstant("OUTPUT_TILED_ORDER", GetTiledOutputOrder(output_ndims)));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_SLICE_NUM", CeilDiv(f, fsv_alignment)));
    jit.AddConstant(MakeJitConstant("TILE_SIZE", tile_size));
    jit.AddConstant(MakeJitConstant("DEFAULT_TILE_SIZE", DEFAULT_TILE_SIZE));
    jit.AddConstant(MakeJitConstant("FSV_ALIGNMENT", fsv_alignment));
    jit.AddConstant(MakeJitConstant("DEFAULT_STRIDE", 16));

    // whether F is aligned
    if (f % fsv_alignment != 0) {
        jit.AddConstant(MakeJitConstant("F_REMAINDER_SIZE", f % fsv_alignment));
        jit.AddConstant(MakeJitConstant("F_REMAINDER_CONDITION", "(f >= (INPUT0_FEATURE_NUM - F_REMAINDER_SIZE)) && (f < INPUT0_FEATURE_NUM)"));
        jit.AddConstant(MakeJitConstant("F_NO_REMAINDER_CONDITION", "(f < (INPUT0_FEATURE_NUM - F_REMAINDER_SIZE))"));
    } else {
        jit.AddConstant(MakeJitConstant("F_NO_REMAINDER_CONDITION", "(f < INPUT0_FEATURE_NUM)"));
    }

    // whether x is tile_size-aligned
    if (x % DEFAULT_TILE_SIZE != 0) {
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE", x % DEFAULT_TILE_SIZE));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_CONDITION", "(x >= (INPUT0_SIZE_X - X_REMAINDER_SIZE)) && (x < INPUT0_SIZE_X)"));
    }

    return jit;
}

KernelsData ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    return GetCommonKernelsData(orgParams);
}

bool ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx::Validate(const Params& p) const {
    if (!ReorderKernelBase::Validate(p)) {
        return false;
    }

    const reorder_params& params = static_cast<const reorder_params&>(p);
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    // decreamental-dims are not supported
    if (input.GetDims().size() > output.GetDims().size()) {
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

KernelsPriority ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx::GetKernelsPriority(const Params& p) const {
    const reorder_params& params = static_cast<const reorder_params&>(p);
    const auto& input = params.inputs[0];

    const size_t f = input.Feature().v;
    const size_t x = input.X().v;

    const size_t tile_size = GetTileSize(params);
    const size_t fsv_alignment = GetFsvAlignment(params);

    if (f <= fsv_alignment && x < tile_size) {
        return DONT_USE_IF_HAVE_SOMETHING_ELSE;
    }

    return FORCE_PRIORITY_3;
}
}  // namespace kernel_selector
