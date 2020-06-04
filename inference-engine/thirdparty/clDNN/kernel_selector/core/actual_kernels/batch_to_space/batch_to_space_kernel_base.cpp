/*
// Copyright (c) 2020 Intel Corporation
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

#include "batch_to_space_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

bool BatchToSpaceKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::BATCH_TO_SPACE ||
        o.GetType() != KernelType::BATCH_TO_SPACE) {
        return false;
    }

    return true;
}

CommonDispatchData BatchToSpaceKernelBase::SetDefault(const batch_to_space_params& params, const optional_params&) const {
    CommonDispatchData runInfo;

    std::vector<size_t> global = { params.output.Batch().v,
                                   params.output.Feature().v,
                                   params.output.W().v * params.output.Z().v * params.output.Y().v * params.output.X().v };

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

JitConstants BatchToSpaceKernelBase::GetJitConstants(const batch_to_space_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto makeJitConstForParam = [](JitConstants& jit, const std::string name, const std::vector<int32_t> vec, const size_t default_value) {
        jit.AddConstant(MakeJitConstant(name + "_SIZES", vec));
        jit.AddConstant(MakeJitConstant(name + "_BATCH", vec[0]));
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", vec[1]));

        switch(vec.size()) {
            case 4: //BFYX
                jit.AddConstant(MakeJitConstant(name + "_W", default_value));
                jit.AddConstant(MakeJitConstant(name + "_Z", default_value));
                jit.AddConstant(MakeJitConstant(name + "_Y", vec[2]));
                jit.AddConstant(MakeJitConstant(name + "_X", vec[3]));
            break;
            case 5: //BFZYX
                jit.AddConstant(MakeJitConstant(name + "_W", default_value));
                jit.AddConstant(MakeJitConstant(name + "_Z", vec[2]));
                jit.AddConstant(MakeJitConstant(name + "_Y", vec[3]));
                jit.AddConstant(MakeJitConstant(name + "_X", vec[4]));
            break;
            case 6: //BFWZYX
                jit.AddConstant(MakeJitConstant(name + "_W", vec[2]));
                jit.AddConstant(MakeJitConstant(name + "_Z", vec[3]));
                jit.AddConstant(MakeJitConstant(name + "_Y", vec[4]));
                jit.AddConstant(MakeJitConstant(name + "_X", vec[5]));
            break;
        }
    };

    makeJitConstForParam(jit, "BLOCK_SHAPE", params.bts_params[0], 1);
    makeJitConstForParam(jit, "CROPS_BEGIN", params.bts_params[1], 0);
    makeJitConstForParam(jit, "CROPS_END", params.bts_params[2], 0);

    return jit;
}

KernelsData BatchToSpaceKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, float estimatedTime) const {
    KernelData kd = KernelData::Default<batch_to_space_params>(params);
    batch_to_space_params& newParams = *static_cast<batch_to_space_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto runInfo = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = estimatedTime;

    return { kd };
}
}  // namespace kernel_selector
