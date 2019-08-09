/*
// Copyright (c) 2019 Intel Corporation
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
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData ShuffleChannelsKernelRef::SetDefault(const shuffle_channels_params& params,
                                                        const optional_params&) const {
    CommonDispatchData runInfo;

    std::vector<size_t> global = {params.output.Batch().v,
                                  params.output.Feature().v,
                                  params.output.Y().v * params.output.X().v};

    auto local = GetOptimalLocalWorkGroupSizes(global);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
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
    KernelData kd = KernelData::Default<shuffle_channels_params>(params);
    shuffle_channels_params& newParams = *static_cast<shuffle_channels_params*>(kd.params.get());

    assert(params.GetType() == KernelType::SHUFFLE_CHANNELS);

    auto runInfo = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
