// Copyright (c) 2016-2020 Intel Corporation
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


#include <iostream>
#include "tensor_type.h"
#include "concatenation_kernel_base.h"
#include <algorithm>
#include <vector>

namespace kernel_selector {
Tensor::DataChannelName ConcatenationKernelBase::GetConcatChannel(const concatenation_params& params) const {
    switch (params.axis) {
        case ConcatAxis::X:
            return Tensor::DataChannelName::X;
        case ConcatAxis::Y:
            return Tensor::DataChannelName::Y;
        case ConcatAxis::Z:
            return Tensor::DataChannelName::Z;
        case ConcatAxis::W:
            return Tensor::DataChannelName::W;
        case ConcatAxis::FEATURE:
            return Tensor::DataChannelName::FEATURE;
        case ConcatAxis::BATCH:
            return Tensor::DataChannelName::BATCH;
        default:
            return Tensor::DataChannelName::X;
    }
}

int32_t ConcatenationKernelBase::GetConcatChannelIndex(const concatenation_params& params) const {
    return DataTensor::Channelndex(params.output.GetLayout(), GetConcatChannel(params));
}

bool ConcatenationKernelBase::Validate(const Params& p, const optional_params&) const {
    if (p.GetType() != KernelType::CONCATENATION) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    if (GetConcatChannelIndex(params) == -1) {
        return false;
    }

    return true;
}

JitConstants ConcatenationKernelBase::GetJitConstants(const concatenation_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("CONCAT_" + toString(params.axis), 1),
    });

    jit.AddConstant(MakeJitConstant("CONCAT_AXIS_INDEX", GetConcatChannelIndex(params)));
    return jit;
}

ConcatenationKernelBase::DispatchData ConcatenationKernelBase::SetDefault(const concatenation_params& params) const {
    DispatchData kd;

    const auto& dims = params.inputs[0].GetDims();
    auto layout = params.inputs[0].GetLayout();

    std::vector<int> idx = { DataTensor::Channelndex(layout, Tensor::DataChannelName::BATCH),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::FEATURE),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::Y),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::X) };

    // Determine global work sizes.
    kd.gws0 = idx[2] != -1 ? dims[idx[2]].v : 1;  // Y
    kd.gws1 = idx[1] != -1 ? dims[idx[1]].v : 1;  // F
    kd.gws2 = idx[0] != -1 ? dims[idx[0]].v : 1;  // B

    kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
    while (kd.gws0 % kd.lws0 != 0) {
        --kd.lws0;
    }

    kd.lws1 = 1;
    kd.lws2 = 1;
    kd.efficiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    return kd;
}

KernelsData ConcatenationKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const concatenation_params& orgParams = static_cast<const concatenation_params&>(params);

    KernelData kd = KernelData::Default<concatenation_params>(params, orgParams.inputs.size());

    uint32_t lastOffset = 0;
    const auto concatChannelIndex = GetConcatChannelIndex(orgParams);
    float efficiency = FORCE_PRIORITY_1;
    size_t ifm_offset = 0;
    for (size_t i = 0; i < orgParams.inputs.size(); i++) {
        const auto& input = orgParams.inputs[i];

        auto newParams = orgParams;
        newParams.inputs.resize(1);
        newParams.inputs[0] = input;
        size_t ifm = input.Feature().v;
        newParams.isAligned = ifm_offset % 16 == 0 && ifm % 16 == 0;
        ifm_offset += ifm;

        auto& kernel = kd.kernels[i];
        DispatchData runInfo = SetDefault(newParams);
        auto cldnnJit = GetJitConstants(newParams);
        auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        kernel.workGroups.global = {runInfo.gws0, runInfo.gws1, runInfo.gws2};
        kernel.workGroups.local = {runInfo.lws0, runInfo.lws1, runInfo.lws2};
        kernel.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, (uint32_t)i});
        kernel.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        ScalarDescriptor s;
        s.t = ScalarDescriptor::Types::UINT32;
        s.v.u32 = lastOffset;
        kernel.scalars.push_back(s);
        kernel.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

        lastOffset += (uint32_t)input.GetDims()[concatChannelIndex].v;
        efficiency = std::max(efficiency, runInfo.efficiency);
    }

    kd.estimatedTime = efficiency;

    return {kd};
}
}  // namespace kernel_selector
