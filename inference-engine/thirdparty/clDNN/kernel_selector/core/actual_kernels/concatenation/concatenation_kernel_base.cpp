/*
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
*/

#include "tensor_type.h"
#include "concatenation_kernel_base.h"

namespace kernel_selector 
{
    static int32_t GetConcatChannelIndex(const concatenation_params& params)
    {
        Tensor::DataChannelName name = Tensor::DataChannelName::X;
        switch (params.axis)
        {
        case ConcatAxis::X:         name = Tensor::DataChannelName::X; break;
        case ConcatAxis::Y:         name = Tensor::DataChannelName::Y; break;
        case ConcatAxis::FEATURE:   name = Tensor::DataChannelName::FEATURE; break;
        case ConcatAxis::BATCH:     name = Tensor::DataChannelName::BATCH; break;
        default: break;
        }

        return DataTensor::Channelndex(params.output.GetLayout(), name);
    }

    bool ConcatenationKernelBase::Validate(const Params& p, const optional_params&) const
    {
        if (p.GetType() != KernelType::CONCATENATION)
        {
            return false;
        }

        const concatenation_params& params = static_cast<const concatenation_params&>(p);

        if (GetConcatChannelIndex(params) == -1)
        {
            return false;
        }

        return true; 
    }

    JitConstants ConcatenationKernelBase::GetJitConstants(const concatenation_params& params) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        jit.AddConstants({
            MakeJitConstant("CONCAT_" + toString(params.axis), 1),
        });

        jit.AddConstant(MakeJitConstant("CONCAT_AXIS_INDEX", GetConcatChannelIndex(params)));
        return jit;
    }

    ConcatenationKernelBase::DispatchData ConcatenationKernelBase::SetDefault(const concatenation_params& params) const
    {
        DispatchData kd;

        const auto& dims = params.inputs[0].GetDims();
        // Determine global work sizes.
        if (params.inputs[0].GetLayout() != params.output.GetLayout())
        {
            kd.gws0 = dims.size() < 2 ? 1 : dims[2].v;
            kd.gws1 = dims.size() < 3 ? 1 : dims[1].v;
            kd.gws2 = dims.size() < 4 ? 1 : dims[0].v;                         
        }
        else
        {
            kd.gws0 = dims.size() < 2 ? 1 : dims[1].v;
            kd.gws1 = dims.size() < 3 ? 1 : dims[2].v;
            kd.gws2 = dims.size() < 4 ? 1 : dims[3].v;
        }

        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }

    KernelsData ConcatenationKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const
    {
        if (!Validate(params,  options))
        {
            return{};
        }

        const concatenation_params& orgParams = static_cast<const concatenation_params&>(params);

        KernelData kd = KernelData::Default<concatenation_params>(params, orgParams.inputs.size());

        uint32_t lastOffset = 0;
        const auto concatChannelIndex = GetConcatChannelIndex(orgParams);
        float effiency = FORCE_PRIORITY_1;
        for (size_t i = 0 ; i < orgParams.inputs.size(); i++)
        {
            const auto& input = orgParams.inputs[i];

            auto newParams = orgParams;
            newParams.inputs.resize(1);
            newParams.inputs[0] = input;

            auto& kernel = kd.kernels[i];
            DispatchData runInfo = SetDefault(newParams);
            auto cldnnJit = GetJitConstants(newParams);
            auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, options);
            auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

            kernel.workGroups.global = { runInfo.gws0, runInfo.gws1, runInfo.gws2 };
            kernel.workGroups.local = { runInfo.lws0, runInfo.lws1, runInfo.lws2 };
            kernel.kernelString = GetKernelString(kernelName, jit, entryPoint);
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, (uint32_t)i });
            kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });

            ScalarDescriptor s;
            s.t = ScalarDescriptor::Types::UINT32;
            s.v.u32 = lastOffset;
            kernel.scalars.push_back(s);
            kernel.arguments.push_back({ ArgumentDescriptor::Types::SCALAR, 0 });

            lastOffset += (uint32_t)input.GetDims()[concatChannelIndex].v;
            effiency = std::max(effiency, runInfo.effiency);
        }

        kd.estimatedTime = effiency;

        return{ kd };
    }
}