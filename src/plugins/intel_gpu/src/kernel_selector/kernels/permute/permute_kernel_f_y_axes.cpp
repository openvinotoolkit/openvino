// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_f_y_axes.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "common_tools.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

constexpr size_t cSimpleMemCopyOpDivider = 4UL;
constexpr size_t c3DTransposeBufHeight = 4UL;

size_t GetDivisor(const size_t input_size) {
    std::vector<size_t> v = {/*32,*/ 16, 8, 4, 2, 1};
    auto is_divided = [input_size](size_t i) {
        return input_size % i == 0;
    };
    auto result = std::find_if(begin(v), end(v), is_divided);
    return *result;
}

bool IsSimpleMemCopyOperation(const permute_params& params) {
    return params.inputs[0].X().v > 1 && params.inputs[0].GetLayout() == DataLayout::bfyx;
}

bool Is3DTranspose(const permute_params& params) {
    return params.inputs[0].X().v > 1 && params.inputs[0].GetLayout() != DataLayout::bfyx;
}

size_t GetFeatureBlockSize(const permute_params& params) {
    return std::min(GetDivisor(params.inputs[0].Feature().v), GetDivisor(params.inputs[0].Y().v));
}

size_t GetTileHeight(const permute_params& params) {
    size_t min_divisor{};
    if (params.inputs[0].X().v == 1) {
        min_divisor = std::min(GetDivisor(params.inputs[0].Feature().v), GetDivisor(params.inputs[0].Y().v));
    } else {
        min_divisor = std::min({GetDivisor(params.inputs[0].Feature().v),
                                GetDivisor(params.inputs[0].Y().v),
                                GetDivisor(params.inputs[0].X().v)});
    }
    if (Is3DTranspose(params)) {
        return std::min(min_divisor, c3DTransposeBufHeight);
    }
    return min_divisor;
}

size_t GetTileWidth(const permute_params& params) {
    const Datatype input_type = params.inputs[0].GetDType();
    const Datatype output_type = params.outputs[0].GetDType();

    size_t min_divisor = GetTileHeight(params);
    if (IsSimpleMemCopyOperation(params)) {
        min_divisor = std::min(min_divisor, cSimpleMemCopyOpDivider);
    }

    // i64 only supports tile size 4
    if ((input_type == Datatype::INT64) || (output_type == Datatype::INT64)) {
        min_divisor = min_divisor >= 4 ? min_divisor / 2 : min_divisor;
    }
    if (input_type == Datatype::F16) {
        min_divisor = min_divisor * 2;
    }
    if (input_type == Datatype::INT8 || input_type == Datatype::UINT8) {
        min_divisor = min_divisor * 4;
    }

    if (params.inputs[0].X().v == 1) {
        return std::min(params.inputs[0].Y().v, min_divisor);
    }
    return std::min(GetDivisor(params.inputs[0].X().v), min_divisor);
}

size_t GetTileSize(const permute_params& params) {
    return std::min(GetTileHeight(params), GetTileWidth(params));
}

}  // namespace

ParamsKey PermuteKernel_f_y_axes::GetSupportedKey() const {
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
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants PermuteKernel_f_y_axes::GetJitConstants(const permute_params& params,
                                                     const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (params.inputs[0].X().v != 1) {
        if (IsSimpleMemCopyOperation(params)) {
            jit.AddConstant(MakeJitConstant("PERMUTE_SIMPLE_MEM_COPY", ""));
        }
        if (Is3DTranspose(params)) {
            jit.AddConstant(MakeJitConstant("THREE_DIM_TRANSPOSE", ""));
        }
    }

    const size_t tile_width = GetTileWidth(params);
    const size_t tile_size = GetTileSize(params);
    const size_t vector_size = IsSimpleMemCopyOperation(params) ? std::min(tile_width, static_cast<size_t>(4)): std::min(tile_size, static_cast<size_t>(4));
    const size_t j_times = IsSimpleMemCopyOperation(params) ? tile_width / vector_size : tile_size / vector_size;
    const size_t feature_block_size = GetFeatureBlockSize(params);
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", tile_width));
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vector_size));
    jit.AddConstant(MakeJitConstant("J_TIMES", j_times));
    jit.AddConstant(MakeJitConstant("TILE_SIZE", tile_size));
    jit.AddConstant(MakeJitConstant("FEATURE_BLOCK_SIZE", feature_block_size));

    const auto layout = params.inputs.front().GetLayout();
    if (!SimpleLayout(layout)) {
        const auto subgroup_size = Is3DTranspose(params) ? feature_block_size : tile_size;
        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subgroup_size));
    }

    jit.Merge(MakeTypeJitConstants(params.inputs[0].GetDType(), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        const std::vector<std::string> original_output_order = {"b_idx", "f_out_idx", "y_out_idx", "x_idx"};
        const FusedOpsConfiguration conf_scalar = {"", original_output_order, "res", params.inputs[0].GetDType(), 1};
        Tensor::DataChannelName channel = (IsSimpleMemCopyOperation(params) || Is3DTranspose(params)) ? Tensor::DataChannelName::X \
                                          : Tensor::DataChannelName::Y;
        const FusedOpsConfiguration conf_vec = {"_VEC", original_output_order, "res", params.inputs[0].GetDType(), vector_size, LoadType::LT_UNALIGNED, \
                                                BoundaryCheck::ENABLED, IndexType::TENSOR_COORD, channel};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar, conf_vec}));
    }
    return jit;
}

static inline std::vector<size_t> GetGWS(const permute_params& params) {
    const auto& in = params.inputs[0];
    std::vector<size_t> gws;
    auto block_size = IsSimpleMemCopyOperation(params) ? GetTileWidth(params) : GetTileSize(params);
    if (params.inputs[0].X().v == 1) {
        gws = {in.X().v, in.Y().v / block_size, (in.Batch().v * in.Feature().v)};
    } else {
        if (Is3DTranspose(params)) {
            gws = {in.X().v / block_size, in.Y().v / GetFeatureBlockSize(params), (in.Batch().v * in.Feature().v)};
        } else {
            gws = {in.X().v / block_size, in.Y().v, (in.Batch().v * in.Feature().v)};
        }
    }
    return gws;
}

CommonDispatchData PermuteKernel_f_y_axes::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    dispatchData.gws = GetGWS(params);
    if (IsSimpleMemCopyOperation(params)) {
        auto in_layout = params.inputs[0].GetLayout();
        auto out_layout = params.outputs[0].GetLayout();
        const std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {
            {Tensor::DataChannelName::X},
            {Tensor::DataChannelName::Y},
            {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        dispatchData.lws =
            GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    } else if (Is3DTranspose(params)) {
        dispatchData.lws = {1, 1, GetFeatureBlockSize(params)};
    } else {
        dispatchData.lws = {1, 1, GetTileSize(params)};
    }
    return dispatchData;
}

bool PermuteKernel_f_y_axes::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        return false;
    }

    const auto is_swapping_f_with_y = [](const std::vector<uint16_t>& order) {
        // Target transform: Swap feature with y
        // IE order:    0 2 1 3 => bfyx -> byfx
        // cldnn order: 0 3 2 1 => bfxy -> byxf
        if (order.size() != 4) {
            return false;
        }
        if (order[0] != 0 || order[1] != 3 || order[2] != 2 || order[3] != 1) {
            return false;
        }
        return true;
    };

    const auto& params = dynamic_cast<const permute_params&>(p);
    const auto& in = params.inputs[0];
    const auto in_layout = in.GetLayout();
    const auto& out = params.outputs[0];
    const auto& out_layout = out.GetLayout();

    const auto feature_div = GetDivisor(in.Feature().v);
    const auto y_div = GetDivisor(in.Y().v);
    if (feature_div == 1 || y_div == 1) {
        return false;
    }
    if (in.X().v > 1 && GetDivisor(in.X().v) == 1) {
        return false;
    }
    if (!is_swapping_f_with_y(params.order)) {
        return false;
    }

    if (in_layout != out_layout) {
        return false;
    }

    // Accept only supported blocked layouts and SIMD sizes.
    if (!SimpleLayout(in_layout)) {
        const auto feature_block_size = GetFeatureBlockSize(params);
        const auto tile_size = GetTileSize(params);
        const auto subgroup_size = Is3DTranspose(params) ? feature_block_size : tile_size;
        if (!(IsSIMDSizeSupported(params.engineInfo, subgroup_size) &&
              (in_layout == DataLayout::b_fs_yx_fsv32 || in_layout == DataLayout::b_fs_yx_fsv16))) {
            return false;
        }
    }

    return true;
}

KernelsPriority PermuteKernel_f_y_axes::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}
}  // namespace kernel_selector
