/*
// Copyright (c) 2019-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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

bool ShuffleChannelsKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::SHUFFLE_CHANNELS ||
        o.GetType() != KernelType::SHUFFLE_CHANNELS) {
        return false;
    }

    const shuffle_channels_params& params = static_cast<const shuffle_channels_params&>(p);

    if (params.inputs[0].Dimentions() > 4)
        return false;

    return true;
}

CommonDispatchData ShuffleChannelsKernelRef::SetDefault(const shuffle_channels_params& params,
                                                        const optional_params&) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = { params.output.Batch().v,
                         params.output.Feature().v,
                         params.output.Y().v * params.output.X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

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

KernelsData ShuffleChannelsKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<shuffle_channels_params>(params);
    shuffle_channels_params& newParams = *static_cast<shuffle_channels_params*>(kd.params.get());

    assert(params.GetType() == KernelType::SHUFFLE_CHANNELS);

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
