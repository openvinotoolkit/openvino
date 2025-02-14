// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_sequence_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey ReverseSequenceKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData ReverseSequenceKernelRef::SetDefault(const reverse_sequence_params& params) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y }};

    dispatchData.gws = { params.outputs[0].Batch().v,
                         params.outputs[0].Feature().v,
                         params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

JitConstants ReverseSequenceKernelRef::GetJitConstants(const reverse_sequence_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("SEQ_AXIS", params.seq_axis));
    jit.AddConstant(MakeJitConstant("BATCH_AXIS", params.batch_axis));

    return jit;
}

KernelsData ReverseSequenceKernelRef::GetKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<reverse_sequence_params>(params);
    reverse_sequence_params& newParams = *static_cast<reverse_sequence_params*>(kd.params.get());

    assert(params.GetType() == KernelType::REVERSE_SEQUENCE);

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2);

    return {kd};
}

KernelsPriority ReverseSequenceKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
