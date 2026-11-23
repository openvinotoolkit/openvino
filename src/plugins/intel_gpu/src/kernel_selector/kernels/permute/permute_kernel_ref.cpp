// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace kernel_selector {
namespace {
bool Process_F_First(const permute_params& params) {
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();

    std::function<bool(const std::vector<uint16_t>&)> f_to_x = [](const std::vector<uint16_t>& order) {
        return ((int32_t) order[2] == 1);
    };

    std::function<bool(const DataLayout&)> is_fsv = [](const DataLayout& layout) {
        switch (layout) {
            case Tensor::DataLayout::b_fs_yx_fsv4:
            case Tensor::DataLayout::b_fs_yx_fsv16:
            case Tensor::DataLayout::b_fs_yx_fsv32:
            case Tensor::DataLayout::b_fs_zyx_fsv16:
            case Tensor::DataLayout::b_fs_zyx_fsv32:
                return true;
            default:
                return false;
        }
        return false;
    };

    return (f_to_x(params.order) && is_fsv(in_layout) && SimpleLayout(out_layout));
}

// Wider per-work-item vectors make neighbouring lanes stride apart and lose coalescing;
// 8 bytes measured best for window-partition permutes on Xe2.
constexpr size_t kInnerVecMaxBytes = 8;

// Vector width for permutes of planar 4D-6D tensors that keep the innermost axis in place
// (cldnn order[2] == 2, e.g. window partition/reverse): each work item then copies VEC_SIZE
// consecutive X elements with vector loads/stores. Returns 0 when the scalar path must be used.
// Shape-agnostic kernels are excluded so that runtime dispatch updates stay scalar as well.
size_t GetInnerVecSize(const permute_params& params) {
    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];
    if (params.has_dynamic_tensors() || params.is_shape_agnostic || !params.fused_ops.empty() || !params.activations.empty()) {
        return 0;
    }
    auto is_planar = [](DataLayout l) {
        return l == DataLayout::bfyx || l == DataLayout::bfzyx || l == DataLayout::bfwzyx;
    };
    if (!is_planar(in.GetLayout()) || !is_planar(out.GetLayout()) || in.GetDType() != out.GetDType()) {
        return 0;
    }
    // A fused following reorder may change the (planar) output rank: the output is then
    // addressed by the linear offset of the permuted shape.
    if (in.LogicalSize() != out.LogicalSize() || in.PitchesDifferFromLogicalDims() || out.PitchesDifferFromLogicalDims()) {
        return 0;
    }
    // The linear output offset is computed in 32-bit uint arithmetic (size_t measured ~4% slower
    // on Xe2), so larger tensors keep the scalar path.
    if (out.LogicalSize() > std::numeric_limits<uint32_t>::max()) {
        return 0;
    }
    const auto& order = params.order;
    if (order.size() != in.GetDims().size() || order.size() < 4 || order.size() > 6 || order[2] != 2) {
        return 0;
    }
    bool is_identity = true;
    for (size_t i = 0; i < order.size(); ++i) {
        is_identity &= order[i] == i;
    }
    if (is_identity) {
        return 0;
    }
    const size_t elem_bytes = BytesPerElement(in.GetDType());
    for (const size_t vec : {16, 8, 4, 2}) {
        if (vec * elem_bytes <= kInnerVecMaxBytes && in.X().v % vec == 0) {
            return vec;
        }
    }
    return 0;
}

// Linear offset of the permuted coordinates in the unpadded planar permuted shape.
// permute_out_idx[i] names the input coordinate placed at cldnn position i of the output.
std::string GetLinearOutputIndex(const std::vector<std::string>& permute_out_idx) {
    static const std::map<std::string, std::string> sizes = {{"b", "INPUT0_BATCH_NUM"},
                                                             {"f", "INPUT0_FEATURE_NUM"},
                                                             {"x", "INPUT0_SIZE_X"},
                                                             {"y", "INPUT0_SIZE_Y"},
                                                             {"z", "INPUT0_SIZE_Z"},
                                                             {"w", "INPUT0_SIZE_W"}};
    // Outermost to innermost: b, f, then spatials from the last cldnn position down to x.
    std::vector<std::string> outer_to_inner = {permute_out_idx[0], permute_out_idx[1]};
    for (size_t i = permute_out_idx.size() - 1; i > 1; --i) {
        outer_to_inner.push_back(permute_out_idx[i]);
    }
    std::string expr = outer_to_inner[0];
    for (size_t i = 1; i < outer_to_inner.size(); ++i) {
        expr = "(" + expr + ")*" + sizes.at(outer_to_inner[i]) + "+" + outer_to_inner[i];
    }
    return "OUTPUT_OFFSET+" + expr;
}

void GetOrderVector(std::string s, std::vector<std::string>* res) {
    size_t pos_start = 0, pos_end;
    std::string token;
    while ((pos_end = s.find(",", pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + 1;
        res->push_back(token);
    }

    res->push_back(s.substr(pos_start));
}

std::string GetReorderedOutputOrder(const permute_params& params,
                                    const std::vector<std::string>& permute_out_idx,
                                    const std::pair<size_t, size_t>& dim_change) {
    std::map<std::string, std::string> size_str_map = {
        {"b", "INPUT0_BATCH_NUM"},
        {"f", "INPUT0_FEATURE_NUM"},
        {"w", "INPUT0_SIZE_W"},
        {"z", "INPUT0_SIZE_Z"},
        {"y", "INPUT0_SIZE_Y"},
        {"x", "INPUT0_SIZE_X"}
    };

    int32_t dim_diff = static_cast<int32_t>(dim_change.first) - static_cast<int32_t>(dim_change.second);

    std::string reordered_order = permute_out_idx[0] + "," + permute_out_idx[1] + ",";
    if (dim_diff > 0) {
        // dim is shrinked
        std::vector<std::string> merged_indices;
        if (dim_diff == 2) merged_indices.push_back(permute_out_idx[dim_change.first - 3]);
        merged_indices.push_back(permute_out_idx[dim_change.first - 2]);
        merged_indices.push_back(permute_out_idx[dim_change.first - 1]);
        std::string pitches = "1";
        for (size_t i = 0 ; i < merged_indices.size(); ++i) {
            if (i > 0) reordered_order += "+";
            reordered_order +=  (merged_indices[i] + "*" + pitches);
            pitches = size_str_map[merged_indices[i]] + "*" + pitches;
        }
        for (size_t i = dim_change.first - 1 - merged_indices.size(); i > 1; --i) {
            reordered_order += ((", " + permute_out_idx[i]));
        }
    } else {
        // dim is expanded
        if (dim_change.first == 4 && dim_change.second == 5) {
            reordered_order += (permute_out_idx.back() + "/" + toCodeString(params.outputs[0].Y().v)
                                 + ", " + permute_out_idx.back() + "%" + toCodeString(params.outputs[0].Y().v)
                                 + ", " + permute_out_idx[2]);
        } else if (dim_change.first == 4 && dim_change.second == 6) {
            reordered_order += (permute_out_idx.back() + "/ (" + toCodeString(params.outputs[0].Y().v)
                                 + " * " + toCodeString(params.outputs[0].Z().v) + ")"
                                 + ", " + permute_out_idx.back() + "/" + toCodeString(params.outputs[0].Y().v)
                                 + ", " + permute_out_idx.back() + "%" + toCodeString(params.outputs[0].Y().v)
                                 + ", " + permute_out_idx[2]);
        } else if (dim_change.first == 5 && dim_change.second == 6) {
            reordered_order += (permute_out_idx.back() + "/" + toCodeString(params.outputs[0].Z().v)
                                 + ", " + permute_out_idx.back() + "%" + toCodeString(params.outputs[0].Z().v)
                                 + ", " + permute_out_idx[3]
                                 + ", " + permute_out_idx[2]);
        }
    }
    return reordered_order;
}
}  // namespace

ParamsKey PermuteKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BF16);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::BF16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

CommonDispatchData PermuteKernelRef::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();

    const auto& in =  params.inputs[0];
    if (Process_F_First(params)) {
        // f is contiguous in output
        // if both input and output are blocked format, need to process with f axis only for the blocked size (TODO)
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{Tensor::DataChannelName::FEATURE},
                                                                        {Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                                                                        {Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                         Tensor::DataChannelName::U, Tensor::DataChannelName::V,
                                                                         Tensor::DataChannelName::BATCH}};
        dispatchData.gws = {in.Feature().v, in.X().v * in.Y().v, in.Z().v * in.W().v  * in.U().v  * in.V().v * in.Batch().v};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    } else {
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{Tensor::DataChannelName::X},
                                                                        {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                         Tensor::DataChannelName::U, Tensor::DataChannelName::V},
                                                                        {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        const size_t vec_size = std::max<size_t>(GetInnerVecSize(params), 1);
        dispatchData.gws = {in.X().v / vec_size, in.Y().v * in.Z().v * in.W().v * in.U().v  * in.V().v , in.Feature().v * in.Batch().v};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    }

    return dispatchData;
}

bool PermuteKernelRef::Validate(const Params& p) const {
    if (!Parent::Validate(p)) { DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const permute_params& params = static_cast<const permute_params&>(p);

    auto in_rank = params.inputs[0].GetDims().size();
    auto out_rank = params.outputs[0].GetDims().size();
    if (in_rank != out_rank && (in_rank > 6 || out_rank > 6)) {
       DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

JitConstants PermuteKernelRef::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    std::vector<std::string> in_idx;
    std::vector<std::string> permute_out_idx;

    std::pair<size_t, size_t> dim_change;
    bool reorder_to_different_dim = false;
    std::vector<std::string> reordered_out_idx;
    if (DataTensor::ChannelsCount(params.inputs[0].GetLayout()) != DataTensor::ChannelsCount(params.outputs[0].GetLayout())) {
        // subsequent reorder to differnt dimension is fused
        dim_change = {params.inputs[0].GetDims().size(), params.outputs[0].GetDims().size()};
        reorder_to_different_dim = true;
    }

    switch (DataTensor::ChannelsCount(params.inputs[0].GetLayout())) {
        case 8: in_idx = {"b", "f", "x", "y", "z", "w", "u", "v" }; break;
        case 7: in_idx = {"b", "f", "x", "y", "z", "w", "u" }; break;
        case 6: in_idx = {"b", "f", "x", "y", "z", "w" }; break;
        case 5: in_idx = {"b", "f", "x", "y", "z" }; break;
        default: in_idx = {"b", "f", "x", "y" }; break;
    }

    assert(params.order.size() == in_idx.size());
    for (const auto& o : params.order) {
        permute_out_idx.push_back(in_idx[o]);
    }

    std::string input_order = in_idx[0] + "," + in_idx[1];

    for (size_t i = in_idx.size() - 1; i > 1; i--) {
        input_order += "," + in_idx[i];
    }

    jit.AddConstant(MakeJitConstant("IN_IDX", "INPUT0_GET_INDEX(" + input_order + ")"));
    if (Process_F_First(params)) {
        jit.AddConstant(MakeJitConstant("F_FIRST", 1));
    }
    if (const size_t vec_size = GetInnerVecSize(params); vec_size > 0) {
        jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));
        jit.AddConstant(MakeJitConstant("OUT_IDX", GetLinearOutputIndex(permute_out_idx)));
    } else if (reorder_to_different_dim) {
        auto reordered_order = GetReorderedOutputOrder(params, permute_out_idx, dim_change);
        jit.AddConstant(MakeJitConstant("OUT_IDX", "OUTPUT_GET_INDEX(" + reordered_order + ")"));
        GetOrderVector(reordered_order, &reordered_out_idx);
    } else {
        std::string output_order = permute_out_idx[0] + "," + permute_out_idx[1];
        for (size_t i = in_idx.size() - 1; i > 1; i--) {
           output_order += "," + permute_out_idx[i];
        }
        jit.AddConstant(MakeJitConstant("OUT_IDX", "OUTPUT_GET_INDEX(" + output_order + ")"));
    }

    if (!params.fused_ops.empty()) {
        for (size_t i = 0; i < (permute_out_idx.size() - 2) / 2; i++) {
            std::swap(permute_out_idx[2 + i], permute_out_idx[permute_out_idx.size() - 1 - i]);
        }

        if (reorder_to_different_dim) {
            FusedOpsConfiguration conf = {"", reordered_out_idx, "input_var", params.inputs[0].GetDType(), 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        } else {
            FusedOpsConfiguration conf = {"", permute_out_idx, "input_var", params.inputs[0].GetDType(), 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }
    return jit;
}

KernelsPriority PermuteKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
