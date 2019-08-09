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


#include "activation_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ActivationKernelBase::DispatchData ActivationKernelBase::SetDefault(const activation_params& arg) const {
    const auto& out = arg.output;

    DispatchData runInfo;
    std::vector<size_t> global;
    std::vector<size_t> local;
    if (out.GetLayout() == DataLayout::bfzyx) {
        global = {out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v};
        local = GetOptimalLocalWorkGroupSizes(global);
    } else if (out.GetLayout() == DataLayout::yxfb) {
        global = {out.Feature().v * out.Batch().v, out.X().v, out.Y().v};
        local = GetOptimalLocalWorkGroupSizes(global);
    } else if (out.GetLayout() == DataLayout::bfyx_f16) {
        global = {Align(out.Feature().v, 16) * out.Batch().v, out.X().v, out.Y().v};
        local = {16, 1, 1};
    } else {
        global = {out.X().v, out.Y().v, out.Feature().v * out.Batch().v};
        local = GetOptimalLocalWorkGroupSizes(global);
    }

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];
    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    runInfo.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    runInfo.fp16UnitUsed = out.GetDType() == Datatype::F16;

    return runInfo;
}

JitConstants ActivationKernelBase::GetJitConstants(const activation_params& params, DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& inputNlParams = params.inputActivationParams;

    jit.AddConstants({
        MakeJitConstant("PARAMS_NUM", GetActivationAdditionalParamsNumber(params.activation.function)),
    });

    if (!inputNlParams.empty()) {
        jit.AddConstants({
            MakeJitConstant("ADDITIONAL_PARAMS", inputNlParams[0]),
            MakeJitConstant("PARAMETERIZED", ""),
        });
    }

    return jit;
}

bool ActivationKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::ACTIVATION ||
        o.GetType() != KernelType::ACTIVATION) {
        return false;
    }

    return true;
}

KernelsData ActivationKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<activation_params>(params);

    activation_params& newParams = *static_cast<activation_params*>(kd.params.get());
    const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

    auto runInfo = SetDefault(newParams);
    auto cldnn_jit = GetJitConstants(newParams, runInfo);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    if (newParams.gradient)
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});

    if (!newParams.inputActivationParams.empty()) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::SLOPE, 0});
    }

    kd.estimatedTime = runInfo.effiency;

    return {kd};
}
}  // namespace kernel_selector
