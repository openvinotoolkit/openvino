// Copyright (c) 2016 Intel Corporation
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

#include "upsampling_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
bool UpSamplingKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::UPSAMPLING || o.GetType() != KernelType::UPSAMPLING) {
        return false;
    }

    const upsampling_params& params = static_cast<const upsampling_params&>(p);

    if (params.inputs.size() == 0) {
        return false;
    }

    return true;
}

JitConstants UpSamplingKernelBase::GetJitConstants(const upsampling_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    auto x_ratio = static_cast<float>(input.X().v) / static_cast<float>(output.X().v);
    auto y_ratio = static_cast<float>(input.Y().v) / static_cast<float>(output.Y().v);

    jit.AddConstants({
        MakeJitConstant(toString(params.sampleType), ""),
        MakeJitConstant("X_RATIO", x_ratio),
        MakeJitConstant("Y_RATIO", y_ratio),
    });

    return jit;
}

KernelsData UpSamplingKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<upsampling_params>(params);
    upsampling_params& newParams = *static_cast<upsampling_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    const auto& out = newParams.output;

    kernel.workGroups.global = {out.X().v, out.Y().v, out.Feature().v * out.Batch().v};
    kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector