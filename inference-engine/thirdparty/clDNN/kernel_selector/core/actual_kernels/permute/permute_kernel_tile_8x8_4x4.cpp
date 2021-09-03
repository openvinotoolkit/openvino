// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_tile_8x8_4x4.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <functional>
#include <cmath>

// Tile size : 4x4 or 8x8
#define MIN_TILE_SIZE 4
#define DEFAULT_TILE_SIZE 8

namespace kernel_selector {

ParamsKey PermuteKernel_tile_8x8_4x4::GetSupportedKey() const {
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
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

static inline size_t GetTileSize(const permute_params& params) {
    // supports 4x4 or 8x8 tiling
    if (params.inputs[0].X().v < DEFAULT_TILE_SIZE || params.inputs[0].Feature().v < DEFAULT_TILE_SIZE)
        return MIN_TILE_SIZE;

    if ((params.inputs[0].GetDType() == Datatype::INT64) || (params.output.GetDType() == Datatype::INT64))
        return MIN_TILE_SIZE;

    return DEFAULT_TILE_SIZE;
}

static inline std::vector<std::string> GetFusedOpOrderVector(size_t size) {
    std::vector<std::string> res;
    switch (size) {
        case 4 :
            res = {"b", "y", "(x * TILE_SIZE + i)", "(f * TILE_SIZE + lh)"};
            break;
        case 5 :
            res = {"b", "z", "y", "(x * TILE_SIZE + i)", "(f * TILE_SIZE + lh)"};
            break;
        case 6 :
            res = {"b", "w", "z", "y", "(x * TILE_SIZE + i)", "(f * TILE_SIZE + lh)"};
            break;
        default : throw std::runtime_error("Unsupported combination\n");
    }
    return res;
}

static inline std::string GetTiledOutputOrder(const permute_params& params) {
    std::pair<size_t, size_t> dim_change = {params.inputs[0].GetDims().size(), params.output.GetDims().size()};

    std::string order_str = "";
    int32_t dim_diff = static_cast<int32_t>(dim_change.first) - static_cast<int32_t>(dim_change.second);

    if (dim_diff == 0) {
        switch (dim_change.first) {
            case 4 :
                order_str = "b, y, (x * TILE_SIZE + lh), (f * TILE_SIZE)";
                break;
            case 5 :
                order_str = "b, z, y, (x * TILE_SIZE + lh), (f * TILE_SIZE)";
                break;
            case 6 :
                order_str = "b, w, z, y, (x * TILE_SIZE + lh), (f * TILE_SIZE)";
                break;
            default : throw std::runtime_error("Unsupported combination\n");
        }
    } else if (dim_diff > 0) {
        // dim is shrinked
        order_str = "b, z + lh, y * INPUT0_SIZE_X + x, f";
        if (dim_change.first == 5 && dim_change.second == 4) {
            order_str = "b, z, y * INPUT0_SIZE_X + (x * TILE_SIZE + lh), (f*TILE_SIZE)";
        } else if (dim_change.first == 6 && dim_change.second == 4) {
            order_str = "b, w, z * INPUT0_SIZE_Y * INPUT0_SIZE_X +  y * INPUT0_SIZE_X + (x * TILE_SIZE + lh), (f * TILE_SIZE)";
        } else if (dim_change.first == 6 && dim_change.second == 5) {
            order_str = "b, w, z * INPUT0_SIZE_Y + y, x * TILE_SIZE + lh, (f * TILE_SIZE)";
        }
    } else {
        // dim is expanded
        if (dim_change.first == 4 && dim_change.second == 5) {
            order_str = ("b, y,  (x * TILE_SIZE + lh) / " + std::to_string(params.output.Y().v)
                                 + ", (x * TILE_SIZE +lh) % " + std::to_string(params.output.Y().v)
                                 + ", (f * TILE_SIZE)");
        } else if (dim_change.first == 4 && dim_change.second == 6) {
            order_str = ("b, y, (x * TILE_SIZE + lh) / (" + std::to_string(params.output.Y().v)
                                 + " * " + std::to_string(params.output.Z().v) + ")"
                                 + ", (x * TILE_SIZE + lh) / " + std::to_string(params.output.Y().v)
                                 + ", (x * TILE_SIZE + lh) % " + std::to_string(params.output.Y().v)
                                 + ", (f * TILE_SIZE)");
        } else if (dim_change.first == 5 && dim_change.second == 6) {
            order_str = ("b, z, y /" + std::to_string(params.output.Z().v)
                                 + ", y % " + std::to_string(params.output.Z().v)
                                 + ", (x * TILE_SIZE + lh), (f * TILE_SIZE)");
        } else {
            throw std::runtime_error("Unsupported combination\n");
        }
    }
    return order_str;
}

static inline std::string GetTiledInputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
        case 4 :
            order_str = "b, (f * TILE_SIZE + lh), y, (x * TILE_SIZE)";
            break;
        case 5 :
            order_str = "b, (f * TILE_SIZE + lh), z, y, (x * TILE_SIZE)";
            break;
        case 6 :
            order_str = "b, (f * TILE_SIZE + lh), w, z, y, (x * TILE_SIZE)";
            break;
        default : throw std::runtime_error("Unsupported combination\n");
    }
    return order_str;
}

JitConstants PermuteKernel_tile_8x8_4x4::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    size_t tile_size = GetTileSize(params);
    size_t vector_width = tile_size;
    // Note: this is default mode and different vector width is not being used now.
    uint64_t total_lws = dispatchData.lws[0] * dispatchData.lws[1] * dispatchData.lws[2];
    jit.AddConstant(MakeJitConstant("VEC_WIDTH", vector_width));
    jit.AddConstant(MakeJitConstant("INPUT0_TILED_ORDER", GetTiledInputOrder(params.inputs[0].GetDims().size())));
    jit.AddConstant(MakeJitConstant("OUTPUT_TILED_ORDER", GetTiledOutputOrder(params)));
    jit.AddConstant(MakeJitConstant("TILE_SIZE", tile_size));
    jit.AddConstant(MakeJitConstant("N_VECTORS_IN_TILE", tile_size / vector_width));
    jit.AddConstant(MakeJitConstant("LWS", total_lws));
    jit.AddConstant(MakeJitConstant("NFEATURE_TILES", CeilDiv(params.inputs[0].Feature().v, tile_size)));

    std::string normal_tile_cond = "true";
    std::string x_remainder_cond = "true";
    std::string f_remainder_cond = "true";

    if (params.inputs[0].X().v % tile_size) {
        jit.AddConstant(MakeJitConstant("X_REMAINDER_ITEM", params.inputs[0].X().v / tile_size));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE", params.inputs[0].X().v % tile_size));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE_AS_VECTOR", CeilDiv(params.inputs[0].X().v % tile_size, vector_width)));
        normal_tile_cond += " && (x < X_REMAINDER_ITEM)";
        x_remainder_cond += " && (x == X_REMAINDER_ITEM)";
        f_remainder_cond += " && (x < X_REMAINDER_ITEM)";
    }
    if (params.inputs[0].Feature().v % tile_size) {
        jit.AddConstant(MakeJitConstant("F_REMAINDER_ITEM", params.inputs[0].Feature().v / tile_size));
        jit.AddConstant(MakeJitConstant("F_REMAINDER_SIZE", params.inputs[0].Feature().v % tile_size));
        jit.AddConstant(MakeJitConstant("F_REMAINDER_SIZE_AS_VECTOR", CeilDiv(params.inputs[0].Feature().v % tile_size, vector_width)));
        normal_tile_cond += " && (f < F_REMAINDER_ITEM)";
        x_remainder_cond += " && (f < F_REMAINDER_ITEM)";
        f_remainder_cond += " && (f == F_REMAINDER_ITEM)";
    }

    jit.AddConstant(MakeJitConstant("NORMAL_TILE_CONDITION", normal_tile_cond));
    jit.AddConstant(MakeJitConstant("X_REMAINDER_CONDITION", x_remainder_cond));
    jit.AddConstant(MakeJitConstant("F_REMAINDER_CONDITION", f_remainder_cond));
    jit.AddConstant(MakeJitConstant("INPUTVTYPE", "CAT(INPUT0_TYPE, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("OUTPUTVTYPE", "CAT(OUTPUT_TYPE, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("VLOAD", "CAT(vload, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("VSTORE", "CAT(vstore, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("AS_INPUTVTYPE", "CAT(as_, INPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("AS_OUTPUTVTYPE", "CAT(as_, OUTPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("LOCAL_BUF_STRIDE", (tile_size / vector_width) * tile_size));
    jit.AddConstant(MakeJitConstant("TRANS_BUF_SIZE", (tile_size / vector_width) * tile_size * total_lws) );

    if (!params.fused_ops.empty()) {
        std::vector<std::string> output_order = GetFusedOpOrderVector(params.output.GetDims().size());
        FusedOpsConfiguration conf = {"", output_order, "input_var", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

static std::vector<size_t> GetBestLwsFromGws(const permute_params& params, const std::vector<size_t>& gws, const size_t tile_size) {
    std::vector<size_t> lws{1, 1, 1};
    std::vector<size_t> dims{0, 2, 1};

    // SLM size: elemsize * tile_size * tile_size * work_items <= 64K
    size_t elem_size = params.output.ElementSize();
    size_t max_local_mem_size = params.engineInfo.maxLocalMemSize;
    size_t max_num_work_items = std::min((size_t)256, (size_t)max_local_mem_size / (elem_size * tile_size * tile_size));

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

CommonDispatchData PermuteKernel_tile_8x8_4x4::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in =  params.inputs[0];
    size_t tile_size = GetTileSize(params);
    switch (in.GetLayout()) {
        case DataLayout::bfyx:
            dispatchData.gws = {CeilDiv(in.X().v , tile_size), in.Y().v, CeilDiv(in.Feature().v, tile_size) * in.Batch().v};
            break;
        case DataLayout::bfzyx:
            dispatchData.gws = {CeilDiv(in.X().v , tile_size), in.Y().v * in.Z().v, CeilDiv(in.Feature().v, tile_size) * in.Batch().v};
            break;
        case DataLayout::bfwzyx:
            dispatchData.gws = {CeilDiv(in.X().v , tile_size), in.Y().v * in.Z().v * in.W().v, CeilDiv(in.Feature().v, tile_size) * in.Batch().v};
            break;
        default:
            throw std::runtime_error("Unsupported combination\n");
            break;
    }
    dispatchData.lws = GetBestLwsFromGws(params, dispatchData.gws, tile_size);
    return dispatchData;
}

bool PermuteKernel_tile_8x8_4x4::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) return false;

    std::function<bool(const std::vector<uint16_t>&)> is_rotating_except_batch = [](const std::vector<uint16_t>& order) {
        // Target transform: Rotate feature dim to back to be taken as inner-most axis
        // ex) 0(b), 4(f), 1(z), 2(y), 3(x)
        // ex) 0(b), 3(f), 1(y), 2(x)
        if ((int32_t) order[1] != order.size() - 1) return false;
        if ((int32_t) order[0] != 0) return false;
        for (int32_t i = 2; i < (int32_t) order.size(); ++i) {
            if ((int32_t)order[i] !=  (i - 1)) return false;
        }
        return true;
    };

    const permute_params& params = static_cast<const permute_params&>(p);

    if (!is_rotating_except_batch(params.order)) {
        return false;
    }

    if (params.output.PitchesDifferFromLogicalDims() || params.inputs[0].PitchesDifferFromLogicalDims()) {
        return false;
    }

    return true;
}

KernelsPriority PermuteKernel_tile_8x8_4x4::GetKernelsPriority(const Params& params/*params*/, const optional_params& /*options*/) const {
    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    if ((newParams.inputs[0].Feature().v >= DEFAULT_TILE_SIZE) && (newParams.inputs[0].X().v >= DEFAULT_TILE_SIZE)) {
        return FORCE_PRIORITY_1;
    } else if ((newParams.inputs[0].Feature().v >= DEFAULT_TILE_SIZE) || (newParams.inputs[0].X().v >= DEFAULT_TILE_SIZE)) {
        return FORCE_PRIORITY_2;
    } else {
        return FORCE_PRIORITY_3;
    }
}
}  // namespace kernel_selector
