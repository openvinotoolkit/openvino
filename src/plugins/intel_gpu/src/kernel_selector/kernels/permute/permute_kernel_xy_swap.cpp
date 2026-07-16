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
    const size_t tile_size = PickTileSize(params);
    const size_t elems_per_dim = tile_size / kWgDim;
    jit.AddConstant(MakeJitConstant("TILE_SIZE", tile_size));
    jit.AddConstant(MakeJitConstant("WG_DIM", kWgDim));
    jit.AddConstant(MakeJitConstant("ELEMS_PER_DIM", elems_per_dim));

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
    const size_t tile_size = PickTileSize(params);
    const size_t elems_per_dim = tile_size / kWgDim;
    // One WG covers a TILE_SIZE x TILE_SIZE block; with WG = WG_DIM x WG_DIM,
    // each WI does ELEMS_PER_DIM^2 work, so GWS shrinks accordingly.
    dispatchData.gws = {in.X().v / elems_per_dim, in.Y().v / elems_per_dim, in.Batch().v * in.Feature().v};
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
    // Require X and Y to be tile-aligned for at least one supported tile size.
    if (PickTileSize(params) == 0) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

KernelsPriority PermuteKernel_xy_swap::GetKernelsPriority(const Params& /*params*/) const {
    // Prefer over the reference kernel when this kernel applies.
    return FORCE_PRIORITY_2;
}

}  // namespace kernel_selector
