// Copyright (c) 2018 Intel Corporation
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

#include "softmax_loss_grad_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
JitConstants SoftmaxLossGradKernelBase::GetJitConstants(const softmax_loss_grad_params& params) const {
    return MakeBaseParamsJitConstants(params);
}

CommonDispatchData SoftmaxLossGradKernelBase::SetDefault(const softmax_loss_grad_params& params,
                                                         const optional_params&) const {
    CommonDispatchData runInfo;

    std::vector<size_t> global = {params.output.Batch().v * params.output.X().v, 1, 1};

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    runInfo.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    return runInfo;
}

bool SoftmaxLossGradKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::SOFT_MAX_LOSS_GRAD || o.GetType() != KernelType::SOFT_MAX_LOSS_GRAD) {
        return false;
    }

    return true;
}

KernelsData SoftmaxLossGradKernelBase::GetCommonKernelsData(const Params& params,
                                                            const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const softmax_loss_grad_params& orgParams = static_cast<const softmax_loss_grad_params&>(params);
    KernelData kd = KernelData::Default<softmax_loss_grad_params>(params);

    auto runInfo = SetDefault(orgParams, options);
    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);
    kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});

    kd.estimatedTime = runInfo.effiency;

    return {kd};
}
}  // namespace kernel_selector
