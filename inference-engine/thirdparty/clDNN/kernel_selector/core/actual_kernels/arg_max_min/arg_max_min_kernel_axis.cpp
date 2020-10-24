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

#include <core/common/kernel_selector_utils.h>
#include "arg_max_min_kernel_axis.h"

namespace kernel_selector {

size_t getOperationNumber(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.output.Feature().v * params.output.Z().v * params.output.Y().v * params.output.X().v;
        case ArgMaxMinAxis::FEATURE: return params.output.Batch().v * params.output.Z().v * params.output.Y().v * params.output.X().v;
        case ArgMaxMinAxis::Z: return params.output.Batch().v * params.output.Feature().v * params.output.Y().v * params.output.X().v;
        case ArgMaxMinAxis::Y: return params.output.Batch().v * params.output.Feature().v * params.output.Z().v * params.output.X().v;
        case ArgMaxMinAxis::X: return params.output.Batch().v * params.output.Feature().v * params.output.Z().v * params.output.Y().v;
        default:
            throw std::invalid_argument("Unsupported axis");
    }
}

size_t getSortSize(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.inputs[0].Batch().v;
        case ArgMaxMinAxis::FEATURE: return params.inputs[0].Feature().v;
        case ArgMaxMinAxis::Z: return params.inputs[0].Z().v;
        case ArgMaxMinAxis::Y: return params.inputs[0].Y().v;
        case ArgMaxMinAxis::X: return params.inputs[0].X().v;
        default:
            throw std::invalid_argument("Unsupported axis");
    }
}

ParamsKey ArgMaxMinKernelAxis::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::BATCH);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::X);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Y);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Z);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    return k;
}

bool ArgMaxMinKernelAxis::Validate(const Params& p, const optional_params& o) const {
    if (!ArgMaxMinKernelBase::Validate(p, o)) {
        return false;
    }

    const arg_max_min_params& params = static_cast<const arg_max_min_params&>(p);

    if (params.inputs.size() > 1) {
        if (params.inputs[1].PitchesDifferFromLogicalDims() || params.output.PitchesDifferFromLogicalDims())
            return false;
    }

    return true;
}

KernelsData ArgMaxMinKernelAxis::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }
    const arg_max_min_params& orgParams = static_cast<const arg_max_min_params&>(params);

    size_t sort_size = orgParams.argMaxMinSortType == ArgMaxMinSortType::VALUE ? getSortSize(orgParams) : 1;

    DispatchData dispatchData;
    dispatchData.gws = { Align(getOperationNumber(orgParams), 32), sort_size, 1 };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    KernelData kd = KernelData::Default<arg_max_min_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    if (orgParams.outputs_num == 2) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    }

    kd.estimatedTime = FORCE_PRIORITY_3;

    return {kd};
}

JitConstants ArgMaxMinKernelAxis::GetJitConstants(const arg_max_min_params& params) const {
    auto jit = ArgMaxMinKernelBase::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("OPERATION_NUM", getOperationNumber(params)));
    if (params.argMaxMinSortType == ArgMaxMinSortType::VALUE)
        jit.AddConstant(MakeJitConstant("SORT_BY_VALUE", 1));
    else
        jit.AddConstant(MakeJitConstant("SORT_BY_INDEX", 1));

    if (params.outputs_num == 2) {
        jit.AddConstant(MakeJitConstant("SECOND_OUTPUT_EXIST", 1));
    }

    if (params.values_first)
        jit.AddConstant(MakeJitConstant("TOP_K_ORDER", 1));

    return jit;
}
}  // namespace kernel_selector
