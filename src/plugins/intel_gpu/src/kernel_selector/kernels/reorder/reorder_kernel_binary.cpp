// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_binary.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey ReorderKernelBinary::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ReorderKernelBinary::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    KernelData kd = KernelData::Default<reorder_params>(params);
    reorder_params& newParams = *static_cast<reorder_params*>(kd.params.get());

    const auto& input = newParams.inputs[0];
    jit.AddConstant(MakeJitConstant("ELEMENTS_COUNT", input.LogicalSize()));
    jit.AddConstant(MakeJitConstant("IFM_PACK_SIZE", 32));

    if (input.GetDType() == Datatype::BINARY) {
        jit.AddConstant(MakeJitConstant("BINARY_INPUT", 1));
        jit.AddConstant(MakeJitConstant("INPUT_PACKED_FEATURES_NUM", CeilDiv(input.Feature().v, 16)));
    }

    if (params.outputs[0].GetDType() == Datatype::BINARY) {
        jit.AddConstant(MakeJitConstant("BINARY_OUTPUT", 1));
        jit.AddConstant(MakeJitConstant("OUTPUT_PACKED_FEATURES_NUM", CeilDiv(params.outputs[0].Feature().v, 32)));
    }

    return jit;
}

ReorderKernelBinary::DispatchData ReorderKernelBinary::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y }};

    const auto& input = params.inputs[0];

    dispatchData.gws = { input.Batch().v, CeilDiv(input.Feature().v, 32), input.Y().v * input.X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsData ReorderKernelBinary::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    if (orgParams.inputs[0].GetDType() != Datatype::BINARY &&
        orgParams.outputs[0].GetDType() != Datatype::BINARY)
        return {};

    if (orgParams.inputs[0].GetDType() == Datatype::BINARY &&
        orgParams.inputs[0].GetLayout() != DataLayout::b_fs_yx_32fp)
        return {};

    if (orgParams.outputs[0].GetDType() == Datatype::BINARY &&
        orgParams.outputs[0].GetLayout() != DataLayout::b_fs_yx_32fp)
        return {};

    return GetCommonKernelsData(orgParams, options);
}

KernelsPriority ReorderKernelBinary::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
