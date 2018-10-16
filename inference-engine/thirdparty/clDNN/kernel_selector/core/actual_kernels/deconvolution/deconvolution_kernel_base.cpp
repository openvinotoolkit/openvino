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

#include "deconvolution_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector 
{
    std::string deconvolution_params::to_string() const
    {
        std::stringstream s;

        s << base_params::to_string() << "_";
        if (bias.empty())
        {
            s << "no_bias" << "_";
        }
        else
        {
            s << "bias_size:" << bias[0].PhysicalSize() << "_";
        }
        s << filterSize.x << "_" << filterSize.y << "_";
        s << stride.x << "_" << stride.y << "_";
        s << dilation.x << "_" << dilation.y << "_";
        s << padding.x << "_" << padding.y << "_";
        s << split;

        return s.str();
    }

    JitConstants DeconvolutionKernelBase::GetJitConstants(const deconvolution_params& dp) const
    {
        JitConstants jit = WeightBiasKernelBase::GetJitConstants(dp);
        const auto& padding = dp.padding;
        const auto& input = dp.inputs[0];

        int64_t input_offset_with_padding = (int64_t)input.GetFirstElementOffset() - (dp.filterSize.x - 1 + padding.x)*input.X().pitch - (dp.filterSize.y - 1 + padding.y)*input.Y().pitch;
        input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

        jit.AddConstants({
            MakeJitConstant("STRIDE",                       dp.stride),
            MakeJitConstant("PADDING",                      dp.padding),
            MakeJitConstant("DILATION",                     dp.dilation),
            MakeJitConstant("FILTER_ARRAY_NUM",             dp.split),
            MakeJitConstant("INPUT0_OFFSET_WITH_PADDING",   input_offset_with_padding),
            MakeJitConstant("DEPTHWISE_SEPARABLE_OPT",      dp.depthwiseSeparableOpt),
        });

        return jit;
    }

    DeconvolutionKernelBase::DispatchData DeconvolutionKernelBase::SetDefault(const deconvolution_params& params) const
    {
        auto batch_size = params.output.Batch().v;
        auto output_features = params.output.Feature().v;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;
        size_t gws0 = output_features * batch_size;
        size_t lws0 = std::min(gws0, static_cast<size_t>(32));
        while (gws0 % lws0)
        {
            lws0--;
        }
        kd.gws0 = gws0;
        kd.gws1 = params.output.X().v;
        kd.gws2 = params.output.Y().v;
        kd.lws0 = lws0;
        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }

    KernelsData DeconvolutionKernelBase::GetKernelsData(const Params& params, const optional_params& options) const
    {
        assert(params.GetType() == KernelType::DECONVOLUTION);

        const deconvolution_params& orgParams = static_cast<const deconvolution_params&>(params);

        const std::vector<WeightsLayout> weightsLayouts = {
            WeightsLayout::oiyx,
            WeightsLayout::iyxo,
            WeightsLayout::yxio,
            WeightsLayout::oyxi
        };

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<deconvolution_params>(params);
        deconvolution_params& newParams = *static_cast<deconvolution_params*>(kd.params.get());

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            weightsLayouts,
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, ROUND_ROBIN, true, !orgParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}