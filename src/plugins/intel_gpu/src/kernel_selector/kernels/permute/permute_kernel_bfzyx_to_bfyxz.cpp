// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_bfzyx_to_bfyxz.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <functional>
#include <cmath>

// Tile size : 4x4 or 8x8
#define MIN_TILE_SIZE 4
#define DEFAULT_TILE_SIZE 8

namespace kernel_selector {

ParamsKey PermuteKernel_bfzyx_to_bfyxz::GetSupportedKey() const {
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
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

static inline bool IsMultipleDefaultTileSize(const size_t size) {
    return size % DEFAULT_TILE_SIZE == 0;
}

static inline size_t GetTileSize(const permute_params& params) {
    // supports 4x4 or 8x8 tiling
    if (!IsMultipleDefaultTileSize(params.inputs[0].X().v) || !IsMultipleDefaultTileSize(params.inputs[0].Z().v))
        return MIN_TILE_SIZE;

    if ((params.inputs[0].GetDType() == Datatype::INT64) || (params.outputs[0].GetDType() == Datatype::INT64))
        return MIN_TILE_SIZE;

    return DEFAULT_TILE_SIZE;
}

JitConstants PermuteKernel_bfzyx_to_bfyxz::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    size_t tile_size = GetTileSize(params);
    size_t vector_width = tile_size;
    uint64_t total_lws = dispatchData.lws[0] * dispatchData.lws[1] * dispatchData.lws[2];
    jit.AddConstant(MakeJitConstant("VEC_WIDTH", vector_width));
    jit.AddConstant(MakeJitConstant("TILE_SIZE", tile_size));
    jit.AddConstant(MakeJitConstant("NZ_TILES", CeilDiv(params.inputs[0].Z().v, tile_size)));

    std::string normal_tile_cond = "true";
    std::string x_remainder_cond = "true";
    std::string z_remainder_cond = "true";

    if (params.inputs[0].X().v % tile_size) {
        jit.AddConstant(MakeJitConstant("X_REMAINDER_ITEM", params.inputs[0].X().v / tile_size));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE", params.inputs[0].X().v % tile_size));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE_AS_VECTOR", CeilDiv(params.inputs[0].X().v % tile_size, vector_width)));
        normal_tile_cond += " && (x < X_REMAINDER_ITEM)";
        x_remainder_cond += " && (x == X_REMAINDER_ITEM)";
        z_remainder_cond += " && (x < X_REMAINDER_ITEM)";
    }
    if (params.inputs[0].Z().v % tile_size) {
        jit.AddConstant(MakeJitConstant("Z_REMAINDER_ITEM", params.inputs[0].Z().v / tile_size));
        jit.AddConstant(MakeJitConstant("Z_REMAINDER_SIZE", params.inputs[0].Z().v % tile_size));
        jit.AddConstant(MakeJitConstant("Z_REMAINDER_SIZE_AS_VECTOR", CeilDiv(params.inputs[0].Z().v % tile_size, vector_width)));
        normal_tile_cond += " && (z < Z_REMAINDER_ITEM)";
        x_remainder_cond += " && (z < Z_REMAINDER_ITEM)";
        z_remainder_cond += " && (z == Z_REMAINDER_ITEM)";
    }

    jit.AddConstant(MakeJitConstant("NORMAL_TILE_CONDITION", normal_tile_cond));
    jit.AddConstant(MakeJitConstant("X_REMAINDER_CONDITION", x_remainder_cond));
    jit.AddConstant(MakeJitConstant("Z_REMAINDER_CONDITION", z_remainder_cond));
    jit.AddConstant(MakeJitConstant("INPUTVTYPE", "CAT(INPUT0_TYPE, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("OUTPUTVTYPE", "CAT(OUTPUT_TYPE, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("VLOAD", "CAT(vload, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("VSTORE", "CAT(vstore, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("AS_INPUTVTYPE", "CAT(as_, INPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("AS_OUTPUTVTYPE", "CAT(as_, OUTPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("LOCAL_BUF_STRIDE", (tile_size / vector_width) * tile_size));
    jit.AddConstant(MakeJitConstant("TRANS_BUF_SIZE", (tile_size / vector_width) * tile_size * total_lws) );

    if (!params.fused_ops.empty()) {
        std::vector<std::string> input_order = {"b", "f", "y", "(x * TILE_SIZE + i)", "(z * TILE_SIZE + lh)"};
        FusedOpsConfiguration conf = {"", input_order, "input_var", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

static std::vector<size_t> GetBestLwsFromGws(const permute_params& params, const std::vector<size_t>& gws, const size_t tile_size) {
    std::vector<size_t> lws{1, 1, 1};
    std::vector<size_t> dims{0, 2, 1};

    // SLM size: elemsize * tile_size * tile_size * work_items <= 64K
    const size_t elem_size = params.outputs[0].ElementSize();
    const size_t max_local_mem_size = params.engineInfo.maxLocalMemSize;
    const size_t max_work_group_size = params.engineInfo.maxWorkGroupSize;
    size_t max_num_work_items = std::min(max_work_group_size, max_local_mem_size / (elem_size * tile_size * tile_size));

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

CommonDispatchData PermuteKernel_bfzyx_to_bfyxz::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in =  params.inputs[0];
    size_t tile_size = GetTileSize(params);
    dispatchData.gws = {CeilDiv(in.X().v , tile_size), in.Y().v, CeilDiv(in.Z().v, tile_size) * in.Feature().v * in.Batch().v};
    dispatchData.lws = GetBestLwsFromGws(params, dispatchData.gws, tile_size);
    return dispatchData;
}

bool PermuteKernel_bfzyx_to_bfyxz::Validate(const Params& p) const {
    if (!Parent::Validate(p)) return false;

    std::function<bool(const std::vector<uint16_t>&)> is_rotating_coords = [](const std::vector<uint16_t>& order) {
        const std::vector<uint16_t> expected_order {0, 1, 4, 2, 3};
        if (order.size() != expected_order.size()) return false;
        for (size_t i{}; i < order.size(); ++i)
            if (order[i] != expected_order[i]) return false;
        return true;
    };

    const permute_params& params = static_cast<const permute_params&>(p);

    if (!is_rotating_coords(params.order))
        return false;

    if (params.outputs[0].PitchesDifferFromLogicalDims() || params.inputs[0].PitchesDifferFromLogicalDims())
        return false;

    return true;
}

KernelsPriority PermuteKernel_bfzyx_to_bfyxz::GetKernelsPriority(const Params& params) const {
    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    if (IsMultipleDefaultTileSize(newParams.inputs[0].Z().v) && IsMultipleDefaultTileSize(newParams.inputs[0].X().v)) {
        return FORCE_PRIORITY_1;
    } else if (IsMultipleDefaultTileSize(newParams.inputs[0].Z().v) || IsMultipleDefaultTileSize(newParams.inputs[0].X().v)) {
        return FORCE_PRIORITY_2;
    } else {
        return FORCE_PRIORITY_3;
    }
}
}  // namespace kernel_selector
