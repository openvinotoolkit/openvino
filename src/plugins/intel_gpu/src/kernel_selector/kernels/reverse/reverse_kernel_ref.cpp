// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_kernel_ref.h"

#include <string>
#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReverseKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData ReverseKernelRef::SetDefault(const reverse_params& params) const {
    CommonDispatchData dispatchData;
    auto in = params.inputs[0];
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {
        {Tensor::DataChannelName::BATCH},
        {Tensor::DataChannelName::FEATURE},
        {Tensor::DataChannelName::X, Tensor::DataChannelName::Y}};

    dispatchData.gws = {in.X().v, in.Y().v * in.Z().v * in.W().v, in.Feature().v * in.Batch().v};

    dispatchData.lws =
        GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in.GetLayout(), out_layout, dims_by_gws);

    return dispatchData;
}

namespace {
std::string toString(reverse_mode mode) {
    switch (mode) {
    case reverse_mode::index:
        return "INDEX";
    case reverse_mode::mask:
        return "MASK";
    }
    return "UNKNOWN";
}

}  // namespace
JitConstants ReverseKernelRef::GetJitConstants(const reverse_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant(toString(params.reverseMode) + "_MODE", 1));
    return jit;
}

KernelsData ReverseKernelRef::GetKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<reverse_params>(params);
    reverse_params& newParams = *static_cast<reverse_params*>(kd.params.get());

    assert(params.GetType() == KernelType::REVERSE);

    const auto dispatchData = SetDefault(newParams);
    const auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    const auto cldnn_jit = GetJitConstants(newParams);
    const auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2);

    return {kd};
}

KernelsPriority ReverseKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
