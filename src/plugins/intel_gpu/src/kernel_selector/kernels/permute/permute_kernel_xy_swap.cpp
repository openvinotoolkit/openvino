// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_xy_swap.h"

#include "common_tools.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

// Workgroup tile is square. The WG dimensions are fixed at WG_DIM x WG_DIM to
// keep the SIMD8 scheduling that performed best in our measurements.
// TILE_SIZE may be a multiple of WG_DIM, in which case each WI handles a
// (TILE_SIZE / WG_DIM)^2 sub-block.
constexpr size_t kWgDim = 16;

// Preferred tile sizes, tried in descending order. For 1-byte data types, a
// 64x64 tile keeps SLM usage comparable to the 32x32 tile for f32 and reduces
// the number of workgroups on aligned shapes. The largest tile that divides
// both X and Y is used.
constexpr size_t kTileCandidates[] = {32, 16};
constexpr size_t kOneByteTileCandidates[] = {64, 32, 16};

size_t PickTileSize(const permute_params& params) {
    const auto x = params.inputs[0].X().v;
    const auto y = params.inputs[0].Y().v;

    const auto pick_tile = [x, y](const auto& candidates) -> size_t {
        for (const auto t : candidates) {
            if ((x % t) == 0 && (y % t) == 0)
                return t;
        }
        return 0;  // no tile size applies
    };

    const bool is_one_byte_type = BytesPerElement(params.inputs[0].GetDType()) == 1;
    return is_one_byte_type ? pick_tile(kOneByteTileCandidates) : pick_tile(kTileCandidates);
}

// Tile plan: an exact tile (branch-free fast path) when X and Y are both
// divisible by a candidate tile size, otherwise a WG_DIM tile with per-tile
// remainder handling so arbitrary X/Y (e.g. head_size 72) are still supported.
struct TilePlan {
    size_t tile = 0;
    bool remainder = false;
};

TilePlan PlanTile(const permute_params& params) {
    const size_t exact = PickTileSize(params);
    if (exact != 0)
        return {exact, false};
    // Fall back to a WG_DIM-sized tile and guard the ragged edges. This keeps
    // coalesced loads/stores and SLM transposition for shapes whose X/Y are
    // not tile-aligned, which would otherwise drop to the scalar reference
    // permute (e.g. {0,1,3,2} with Y=72).
    return {kWgDim, true};
}

bool IsXYSwapOrder(const std::vector<uint16_t>& order) {
    // 4D only, last two dims swapped, others identity.
    // cldnn order produced from IE {0,1,3,2} is also {0,1,3,2} for 4D inputs.
    if (order.size() != 4)
        return false;
    return order[0] == 0 && order[1] == 1 && order[2] == 3 && order[3] == 2;
}

}  // namespace

ParamsKey PermuteKernel_xy_swap::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants PermuteKernel_xy_swap::GetJitConstants(const permute_params& params,
                                                    const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    const TilePlan plan = PlanTile(params);
    const size_t tile_size = plan.tile;
    const size_t elems_per_dim = tile_size / kWgDim;
    jit.AddConstant(MakeJitConstant("TILE_SIZE", tile_size));
    jit.AddConstant(MakeJitConstant("WG_DIM", kWgDim));
    jit.AddConstant(MakeJitConstant("ELEMS_PER_DIM", elems_per_dim));
    jit.AddConstant(MakeJitConstant("REMAINDER", plan.remainder ? 1 : 0));

    if (!params.fused_ops.empty()) {
        // Output layout indices for fused ops (bfyx in IE naming).
        const std::vector<std::string> idx_order = {"b", "f", "out_y", "out_x"};
        const FusedOpsConfiguration conf = {"", idx_order, "val", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

CommonDispatchData PermuteKernel_xy_swap::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in = params.inputs[0];
    const TilePlan plan = PlanTile(params);
    const size_t tile_size = plan.tile;
    // One WG covers a TILE_SIZE x TILE_SIZE block; with WG = WG_DIM x WG_DIM,
    // each WI does ELEMS_PER_DIM^2 work, so GWS shrinks accordingly. Round the
    // number of tiles up so ragged edges (REMAINDER path) are still covered.
    const size_t x_tiles = CeilDiv(in.X().v, tile_size);
    const size_t y_tiles = CeilDiv(in.Y().v, tile_size);
    dispatchData.gws = {x_tiles * kWgDim, y_tiles * kWgDim, in.Batch().v * in.Feature().v};
    dispatchData.lws = {kWgDim, kWgDim, 1};
    return dispatchData;
}

bool PermuteKernel_xy_swap::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& params = static_cast<const permute_params&>(p);

    // Only the X<->Y swap pattern on plain bfyx layouts.
    if (!IsXYSwapOrder(params.order)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (params.inputs[0].GetLayout() != DataLayout::bfyx ||
        params.outputs[0].GetLayout() != DataLayout::bfyx) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    // No dynamic shapes for this initial implementation: tile size and divisibility
    // are checked at compile time.
    if (params.has_dynamic_tensors()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (params.inputs[0].PitchesDifferFromLogicalDims() ||
        params.outputs[0].PitchesDifferFromLogicalDims()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    // A WG_DIM tile with remainder handling covers any X/Y, so no divisibility
    // requirement remains; PlanTile always yields a usable tile.

    return true;
}

KernelsPriority PermuteKernel_xy_swap::GetKernelsPriority(const Params& /*params*/) const {
    // Prefer over the reference kernel when this kernel applies.
    return FORCE_PRIORITY_2;
}

}  // namespace kernel_selector
