// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shuffle_channels_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey ShuffleChannelsKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

bool ShuffleChannelsKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SHUFFLE_CHANNELS) {
        return false;
    }

    const shuffle_channels_params& params = static_cast<const shuffle_channels_params&>(p);

    if (params.inputs[0].Dimentions() > 4)
        return false;

    return true;
}

CommonDispatchData ShuffleChannelsKernelRef::SetDefault(const shuffle_channels_params& params) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y }};

    dispatchData.gws = { params.outputs[0].Batch().v,
                         params.outputs[0].Feature().v,
                         params.outputs[0].Y().v * params.outputs[0].X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

JitConstants ShuffleChannelsKernelRef::GetJitConstants(const shuffle_channels_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("GROUPS_NUMBER", params.group));

    auto getDimSizeByAxis = [](const shuffle_channels_params& params) -> size_t {
        switch (params.axis) {
            case 0:
                return params.inputs[0].Batch().v;
            case 1:
                return params.inputs[0].Feature().v;
            case 2:
                return params.inputs[0].Y().v;
            case 3:
                return params.inputs[0].X().v;
        }
        return 0;
    };

    jit.AddConstant(MakeJitConstant("GROUP_SIZE", getDimSizeByAxis(params) / params.group));
    jit.AddConstant(MakeJitConstant("AXIS", params.axis));

    return jit;
}

KernelsData ShuffleChannelsKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<shuffle_channels_params>(params);
    shuffle_channels_params& newParams = *static_cast<shuffle_channels_params*>(kd.params.get());

    assert(params.GetType() == KernelType::SHUFFLE_CHANNELS);

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {kd};
}

KernelsPriority ShuffleChannelsKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
