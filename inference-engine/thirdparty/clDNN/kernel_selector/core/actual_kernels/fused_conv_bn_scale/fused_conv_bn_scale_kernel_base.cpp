/*
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
*/

#include "fused_conv_bn_scale_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector 
{
    bool fused_conv_bn_scale_kernel_base::Validate(const Params& p, const optional_params& o) const
    {
        if (p.GetType() != KernelType::FUSED_CONV_BN_SCALE ||
            o.GetType() != KernelType::FUSED_CONV_BN_SCALE)
        {
            return false;
        }

        const fused_conv_bn_scale_params& params = static_cast<const fused_conv_bn_scale_params&>(p);
        const fused_conv_bn_scale_optional_params& optParams = static_cast<const fused_conv_bn_scale_optional_params&>(o);

        bool bSupportedWeightsLayout = false;

        for (WeightsLayout l : GetSupportedWeightLayouts(params))
        {
            bSupportedWeightsLayout |= params.weights.GetLayout() == l;
        }

        const bool bWeightsOK = bSupportedWeightsLayout || optParams.allowStaticInputReordering;

        return bWeightsOK;
    }

    JitConstants fused_conv_bn_scale_kernel_base::GetJitConstants(const fused_conv_bn_scale_params& params, const DispatchData&) const
    {
        JitConstants mem_consts = WeightBiasKernelBase::GetJitConstants(params);
        const auto& padding = params.padding;
        const auto& input = params.inputs[0];

        int64_t input_offset_with_padding = (int64_t)input.GetFirstElementOffset() - padding.x*input.X().pitch - input.Y().pitch*padding.y;
        input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

        mem_consts.AddConstants({
            MakeJitConstant("STRIDE",                       params.stride),
            MakeJitConstant("PADDING",                      params.padding),
            MakeJitConstant("FILTER_ARRAY_NUM",             params.split),
            MakeJitConstant("DILATION",                     params.dilation),
            MakeJitConstant("INPUT0_OFFSET_WITH_PADDING",   input_offset_with_padding),
            MakeJitConstant("EPSILON", params.epsilon)
        });

        if (params.fused_in_training)
            mem_consts.AddConstant(MakeJitConstant("FUSED_TRAINING", 1));
        if (params.scale_bias)
            mem_consts.AddConstant(MakeJitConstant("SCALE_BIAS_TERM", 1));

        return mem_consts;
    }

    bool fused_conv_bn_scale_kernel_base::CheckWorkGroups(const DispatchData& kd)
    {
        if (kd.gws0 == 0 ||
            kd.gws1 == 0 ||
            kd.gws2 == 0 ||
            kd.lws0 == 0 ||
            kd.lws1 == 0 ||
            kd.lws2 == 0)
        {
            return false;
        }

        if ((kd.gws0 % kd.lws0) != 0 ||
            (kd.gws1 % kd.lws1) != 0 ||
            (kd.gws2 % kd.lws2) != 0)
        {
            return false;
        }

        return true;
    }

    fused_conv_bn_scale_kernel_base::DispatchData fused_conv_bn_scale_kernel_base::SetDefault(const fused_conv_bn_scale_params& params) const
    {
        DispatchData kd;

        const auto& out = params.output;
        kd.fp16UnitUsed = out.GetDType() == Datatype::F16;
        std::vector<size_t> global;
        if (params.output.GetLayout() == DataLayout::bfyx || params.output.GetLayout() == DataLayout::byxf)
        {
            global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        }
        else
        {
            global = { out.Feature().v*out.Batch().v, out.X().v, out.Y().v };
        }

        auto local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }

    KernelsData fused_conv_bn_scale_kernel_base::GetCommonKernelsData(const Params& params, const optional_params& options, float estimated_time) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<fused_conv_bn_scale_params>(params);
        fused_conv_bn_scale_params& newParams = *static_cast<fused_conv_bn_scale_params*>(kd.params.get());

        DispatchData runInfo = SetDefault(newParams);
        
        if (!CheckWorkGroups(runInfo))
        {
            // Internal Error - wrong calculation of global/local work group sizes
            return{};
        }

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            GetSupportedWeightLayouts(newParams),
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto finalKernelName = GetKernelName(newParams);
        auto cldnnJit = GetJitConstants(newParams, runInfo);
        auto entryPoint = GetEntryPoint(finalKernelName, newParams.layerID, options);
        auto jit = CreateJit(finalKernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, params.engineInfo, finalKernelName, jit, entryPoint, "", true, !newParams.bias.empty(), 1);
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });
        uint32_t idx = 1;
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, idx++ });
        if (newParams.scale_bias)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, idx++ });
        if (newParams.fused_in_training)
        {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, idx++ });
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, idx++ });
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, idx });
        }

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}
