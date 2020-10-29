// Copyright (c) 2018-2020 Intel Corporation
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


#include "fully_connected_grad_input_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <algorithm>

namespace kernel_selector {
JitConstants FullyConnectedGradInputKernelBase::GetJitConstants(const fully_connected_grad_input_params& params) const {
    return WeightBiasKernelBase::GetJitConstants(params);
}

FullyConnectedGradInputKernelBase::DispatchData FullyConnectedGradInputKernelBase::SetDefault(
    const fully_connected_grad_input_params& params) const {
    DispatchData kd;

    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;
    size_t gws0 = params.output.Batch().v * params.weights.IFM().v;
    size_t lws0 = std::min(gws0, static_cast<size_t>(32));
    while (gws0 % lws0) {
        lws0--;
    }
    kd.gws0 = gws0;
    kd.gws1 = params.weights.X().v;
    kd.gws2 = params.weights.Y().v;
    kd.lws0 = lws0;
    kd.lws1 = 1;
    kd.lws2 = 1;
    kd.efficiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    return kd;
}

KernelsData FullyConnectedGradInputKernelBase::GetKernelsData(const Params& params,
                                                              const optional_params& options) const {
    assert(params.GetType() == KernelType::FULLY_CONNECTED_GRAD_INPUT);

    const fully_connected_grad_input_params& orgParams = static_cast<const fully_connected_grad_input_params&>(params);

    DispatchData runInfo = SetDefault(orgParams);
    KernelData kd = KernelData::Default<fully_connected_grad_input_params>(params);
    fully_connected_grad_input_params& newParams = *static_cast<fully_connected_grad_input_params*>(kd.params.get());

    bool succeed = UpdateWeightsParams(newParams, options, WeightsLayout::oi, kd.weightsReorderParams);

    if (!succeed) {
        return {};
    }

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     runInfo,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     true,
                     !orgParams.bias.empty());
    kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});

    kd.estimatedTime = runInfo.efficiency;

    return {kd};
}
}  // namespace kernel_selector