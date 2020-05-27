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

#include "cum_sum_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
Tensor::DataChannelName CumSumKernelBase::GetCumSumAxis(const cum_sum_params& params) const {
    switch (params.axis) {
        case CumSumAxis::X:
            return Tensor::DataChannelName::X;
        case CumSumAxis::Y:
            return Tensor::DataChannelName::Y;
        case CumSumAxis::Z:
            return Tensor::DataChannelName::Z;
        case CumSumAxis::W:
            return Tensor::DataChannelName::W;
        case CumSumAxis::FEATURE:
            return Tensor::DataChannelName::FEATURE;
        case CumSumAxis::BATCH:
            return Tensor::DataChannelName::BATCH;
        default:
            return Tensor::DataChannelName::BATCH;
    }
}

int32_t CumSumKernelBase::GetCumSumAxisIndex(const cum_sum_params& params) const {
    return DataTensor::Channelndex(params.output.GetLayout(), GetCumSumAxis(params));
}

size_t CumSumKernelBase::GetRealAxisIndex(const cum_sum_params& params) const {
    size_t index = params.output.Dimentions() - GetCumSumAxisIndex(params) - 1;
    if (params.output.Dimentions() == 6)
        return index;
    else if (params.output.Dimentions() == 5)
        return (index > 1) ? index + 1 : index;
    return (index > 1) ? index + 2 : index;
}

ParamsKey CumSumKernelBase::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants CumSumKernelBase::GetJitConstants(const cum_sum_params& params, DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    if (params.exclusive)
        jit.AddConstant(MakeJitConstant("EXCLUSIVE", 1));
    if (params.reverse)
        jit.AddConstant(MakeJitConstant("REVERSE", 1));
    jit.AddConstant(MakeJitConstant("AXIS", GetRealAxisIndex(params)));

    return jit;
}

CumSumKernelBase::DispatchData CumSumKernelBase::SetDefault(const cum_sum_params& params) const {
    DispatchData runInfo;
    std::vector<size_t> global = {params.output.Batch().v,
                                  params.output.Feature().v * params.output.W().v,
                                  params.output.Z().v * params.output.Y().v * params.output.X().v};

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

KernelsData CumSumKernelBase::GetCommonKernelsData(const Params& params,
                                                   const optional_params& options,
                                                   float estimatedTime) const {
    KernelData kd = KernelData::Default<cum_sum_params>(params);
    cum_sum_params& newParams = *static_cast<cum_sum_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto runInfo = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams, runInfo);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = estimatedTime;

    return {kd};
}

bool CumSumKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::CUM_SUM || o.GetType() != KernelType::CUM_SUM) {
        return false;
    }

    auto& params = static_cast<const cum_sum_params&>(p);
    if (GetCumSumAxisIndex(params) == -1)
        return false;

    if (params.inputs.size() > 1 && params.inputs[1].GetDType() != Datatype::INT32 &&
        params.inputs[1].GetDType() != Datatype::UINT32)
        return false;

    return true;
}

Datatype CumSumKernelBase::GetActivationType(const cum_sum_params& params) const {
    if (params.output.GetDType() == Datatype::F16)
        return Datatype::F16;
    return Datatype::F32;
}
}  // namespace kernel_selector
