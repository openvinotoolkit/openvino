// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
namespace kernel_selector {
ParamsKey PermuteKernelRef::GetSupportedKey() const {
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
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData PermuteKernelRef::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in =  params.inputs[0];

    dispatchData.gws = {in.X().v, in.Y().v * in.Z().v * in.W().v, in.Feature().v * in.Batch().v};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}
bool PermuteKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) return false;

    return true;
}

static void GetOrderVector(std::string s, std::vector<std::string>* res) {
    size_t pos_start = 0, pos_end;
    std::string token;
    while ((pos_end = s.find(",", pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + 1;
        res->push_back(token);
    }

    res->push_back(s.substr(pos_start));
    return;
}

static std::string GetReorderedOutputOrder(const permute_params& params, const std::vector<std::string>& permute_out_idx,
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
            reordered_order += (permute_out_idx.back() + "/" + std::to_string(params.output.Y().v)
                                 + ", " + permute_out_idx.back() + "%" + std::to_string(params.output.Y().v)
                                 + ", " + permute_out_idx[2]);
        } else if (dim_change.first == 4 && dim_change.second == 6) {
            reordered_order += (permute_out_idx.back() + "/ (" + std::to_string(params.output.Y().v)
                                 + " * " + std::to_string(params.output.Z().v) + ")"
                                 + ", " + permute_out_idx.back() + "/" + std::to_string(params.output.Y().v)
                                 + ", " + permute_out_idx.back() + "%" + std::to_string(params.output.Y().v)
                                 + ", " + permute_out_idx[2]);
        } else if (dim_change.first == 5 && dim_change.second == 6) {
            reordered_order += (permute_out_idx.back() + "/" + std::to_string(params.output.Z().v)
                                 + ", " + permute_out_idx.back() + "%" + std::to_string(params.output.Z().v)
                                 + ", " + permute_out_idx[3]
                                 + ", " + permute_out_idx[2]);
        }
    }
    return reordered_order;
}
JitConstants PermuteKernelRef::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    std::vector<std::string> in_idx;
    std::vector<std::string> permute_out_idx;

    std::map<std::string, std::string> size_str_map = {
        {"b", "INPUT0_BATCH_NUM"},
        {"f", "INPUT0_FEATURE_NUM"},
        {"w", "INPUT0_SIZE_W"},
        {"z", "INPUT0_SIZE_Z"},
        {"y", "INPUT0_SIZE_Y"},
        {"x", "INPUT0_SIZE_X"}
    };

    std::pair<size_t, size_t> dim_change;
    bool reorder_to_different_dim = false;
    std::vector<std::string> reordered_out_idx;
    if (DataTensor::ChannelsCount(params.inputs[0].GetLayout()) != DataTensor::ChannelsCount(params.output.GetLayout())) {
        // subsequent reorder to differnt dimension is fused
        dim_change = {params.inputs[0].GetDims().size(), params.output.GetDims().size()};
        reorder_to_different_dim = true;
    }

    switch (DataTensor::ChannelsCount(params.inputs[0].GetLayout())) {
        case 6: in_idx = {"b", "f", "x", "y", "z", "w" }; break;
        case 5: in_idx = {"b", "f", "x", "y", "z" }; break;
        default: in_idx = {"b", "f", "x", "y" }; break;
    }

    assert(params.order.size() == in_idx.size());
    for (auto& o : params.order) {
        permute_out_idx.push_back(in_idx[o]);
    }

    std::string input_order = in_idx[0] + "," + in_idx[1];

    for (size_t i = in_idx.size() - 1; i > 1; i--) {
        input_order += "," + in_idx[i];
    }

    jit.AddConstant(MakeJitConstant("IN_IDX", "INPUT0_GET_INDEX(" + input_order + ")"));

    if (reorder_to_different_dim) {
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
        if (permute_out_idx.size() == 4) {
            std::swap(permute_out_idx[2], permute_out_idx[3]);
        } else if (permute_out_idx.size() == 5) {
            std::swap(permute_out_idx[2], permute_out_idx[4]);
        } else if (permute_out_idx.size() == 6) {
            std::swap(permute_out_idx[2], permute_out_idx[5]);
            std::swap(permute_out_idx[3], permute_out_idx[4]);
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

KernelsPriority PermuteKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
