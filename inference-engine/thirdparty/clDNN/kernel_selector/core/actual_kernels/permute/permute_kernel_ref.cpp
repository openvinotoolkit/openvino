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

    const permute_params& params = static_cast<const permute_params&>(p);

    // currently reorder fusing is supported only for format change, not the layout change
    if (DataTensor::ChannelsCount(params.inputs[0].GetLayout())
        != DataTensor::ChannelsCount(params.output.GetLayout())) {
        return false;
    }

    return true;
}
JitConstants PermuteKernelRef::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    std::vector<std::string> in_idx;
    std::vector<std::string> out_idx;
    switch (DataTensor::ChannelsCount(params.inputs[0].GetLayout())) {
        case 6: in_idx = {"b", "f", "x", "y", "z", "w" }; break;
        case 5: in_idx = {"b", "f", "x", "y", "z" }; break;
        default: in_idx = {"b", "f", "x", "y" }; break;
    }

    assert(params.order.size() == in_idx.size());
    for (auto& o : params.order) {
        out_idx.push_back(in_idx[o]);
    }

    std::string input_order = in_idx[0] + "," + in_idx[1];
    std::string output_order = out_idx[0] + "," + out_idx[1];

    for (size_t i = in_idx.size() - 1; i > 1; i--) {
        input_order += "," + in_idx[i];
        output_order += "," + out_idx[i];
    }

    jit.AddConstant(MakeJitConstant("IN_IDX", "INPUT0_GET_INDEX(" + input_order + ")"));
    jit.AddConstant(MakeJitConstant("OUT_IDX", "OUTPUT_GET_INDEX(" + output_order + ")"));

    if (!params.fused_ops.empty()) {
        if (out_idx.size() == 4) {
            std::swap(out_idx[2], out_idx[3]);
        } else if (out_idx.size() == 5) {
            std::swap(out_idx[2], out_idx[4]);
        } else if (out_idx.size() == 6) {
            std::swap(out_idx[2], out_idx[5]);
            std::swap(out_idx[3], out_idx[4]);
        }

        FusedOpsConfiguration conf = {"", out_idx, "input_var", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

KernelsPriority PermuteKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
