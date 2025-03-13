// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col_to_im_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey ColToImKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData ColToImKernelRef::SetDefault(const col_to_im_params& params) const {
    CommonDispatchData dispatchData;

    // TODO : implement for col_to_im_gpu_ref
    // auto in_layout = params.inputs[0].GetLayout();
    // auto out_layout = params.outputs[0].GetLayout();
    {
        // std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
        //                                                                  { Tensor::DataChannelName::FEATURE },
        //                                                                  { Tensor::DataChannelName::X, Tensor::DataChannelName::Y, Tensor::DataChannelName::Z }};

        // dispatchData.gws = { params.outputs[0].Batch().v,
        //                      params.outputs[0].Feature().v,
        //                      params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v };

        // // The reason why reverse input/output of GetOptimalLocalWorkGroupSizes():
        // // Large X*Y*Z lws size is better than large batch lws, but current GetOptimalLocalWorkGroupSizes not work like that.
        // reverse(dims_by_gws.begin(), dims_by_gws.end());
        // reverse(dispatchData.gws.begin(), dispatchData.gws.end());
        // dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
        // reverse(dispatchData.lws.begin(), dispatchData.lws.end());
        // reverse(dispatchData.gws.begin(), dispatchData.gws.end());

        dispatchData.gws = {1, 1, 1};
        dispatchData.lws = {1, 1, 1};
    }

    return dispatchData;
}

KernelsData ColToImKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority ColToImKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

JitConstants ColToImKernelRef::GetJitConstants(const col_to_im_params& params) const {
    auto jit = Parent::GetJitConstants(params);
    auto input = params.inputs[0];
    auto input_dt = input.GetDType();

    // TODO : implement for col_to_im_gpu_ref
    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (input.Dimentions() == 5) {
            idx_order = {"batch", "feature", "z", "y", "x"};
        } else if (input.Dimentions() == 4) {
            idx_order = {"batch", "feature", "y", "x"};
        }
        FusedOpsConfiguration conf = {"", idx_order, "in_val", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

}  // namespace kernel_selector
