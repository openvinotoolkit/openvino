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

#include "convolution_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector 
{
    bool ConvolutionKernelBase::Validate(const Params& p, const optional_params& o) const
    {
        if (p.GetType() != KernelType::CONVOLUTION ||
            o.GetType() != KernelType::CONVOLUTION)
        {
            return false;
        }

        const convolution_params& params = static_cast<const convolution_params&>(p);
        const convolution_optional_params& optParams = static_cast<const convolution_optional_params&>(o);

        bool bSupportedWeightsLayout = false;

        for (WeightsLayout l : GetSupportedWeightLayouts(params))
        {
            bSupportedWeightsLayout |= params.weights.GetLayout() == l;
        }

        const bool bWeightsOK = bSupportedWeightsLayout || optParams.allowStaticInputReordering;

        if (!bWeightsOK)
        {
            return false;
        }

        return true;
    }

    JitConstants ConvolutionKernelBase::GetJitConstants(const convolution_params& params, const DispatchData& kd) const
    {
        JitConstants mem_consts = WeightBiasKernelBase::GetJitConstants(params);
        const auto& padding = params.padding;
        const auto& input = params.inputs[0];

        int64_t input_offset_with_padding = (int64_t)input.GetFirstElementOffset() - padding.x*input.X().pitch - input.Y().pitch*padding.y;
        input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

        mem_consts.AddConstants({
            MakeJitConstant("STRIDE",                       params.stride),
            MakeJitConstant("PADDING",                      params.padding),
            MakeJitConstant("DILATION",                     params.dilation),
            MakeJitConstant("FILTER_ARRAY_NUM",             params.split),
            MakeJitConstant("INPUT0_OFFSET_WITH_PADDING",   input_offset_with_padding),
            MakeJitConstant("DEPTHWISE_SEPARABLE_OPT",      params.depthwiseSeparableOpt),
            MakeJitConstant("QUANTIZATION_TERM",            params.int8_quantization),
        });

        if (params.int8_quantization)
        {
            mem_consts.AddConstants({ MakeJitConstant("W_QF", params.weights_quantization_factors[0]) });
            mem_consts.AddConstants({ MakeJitConstant("I_QF",params.input_quantization_factor) });

            if (params.output_calibration)
            {
                mem_consts.AddConstant(MakeJitConstant("CALIBRATION_TERM", params.output_calibration));
                mem_consts.AddConstant(MakeJitConstant("O_QF", params.output_calibration_factors[0]));

            }
            else
                mem_consts.AddConstants({ MakeJitConstant("O_QF", params.output_quantization_factor) });
        }

        std::vector<uint32_t> unrollLoopParams{
            params.filterSize.x,
            params.filterSize.y,
            (uint32_t)kd.gemmStyle.globalWorkSizeDX,
            (uint32_t)kd.gemmStyle.globalWorkSizeDY,
            (uint32_t)kd.gemmStyle.globalWorkSizeDZ,
            (uint32_t)kd.gemmStyle.subBlockDimM,
            (uint32_t)kd.gemmStyle.subBlockDimK,
            (uint32_t)kd.gemmStyle.subBlockDimN
        };

        auto loopCount = *std::max_element(unrollLoopParams.begin(), unrollLoopParams.end());

        JitConstants mem_consts_loop = MakeLoopUnrollParamsJitConstants(loopCount);
        mem_consts.Merge(mem_consts_loop);

        return mem_consts;
    }

    bool ConvolutionKernelBase::CheckWorkGroups(const ConvolutionKernelBase::DispatchData& kd)
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

    namespace
    {
        bool CheckTensorForSplit(const DataTensor& t, uint32_t split)
        {
            if (t.PitchesDifferFromLogicalDims())
            {
                auto feature = t.Feature();
                auto featureIndex = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::FEATURE);
                if (featureIndex >= 0 && featureIndex+1 < (int)DataTensor::ChannelsCount(t.GetLayout()))
                {
                    if (feature.v*split <= t.GetDims()[featureIndex+1].pitch)
                    {
                        Tensor::NDims newDims = t.GetDims();
                        newDims[featureIndex].v = feature.v*split;
                        
                        DataTensor newTensor{ newDims, t.GetDType(), t.GetLayout(), t.GetViewOffset(), t.PhysicalSize(), t.GetPaddedVal()};

                        if (newTensor.PitchesDifferFromLogicalDims() == false)
                        {
                            return true;
                        }
                    }
                }

                return false;
            }

            return true;
        }
    }

    bool ConvolutionKernelBase::CheckPitchForSplitOnly(const convolution_params& params)
    {
        // TODO: it's better to add pitch+offset support than handle this case
        return CheckTensorForSplit(params.inputs[0], params.split);
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernelBase::SetDefault(const convolution_params& params, int) const
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

        kd.cldnnStyle.blockWidth = 1;
        kd.cldnnStyle.blockHeight = 1;
        kd.cldnnStyle.prefetch = 0;
        kd.cldnnStyle.inputBlockArraySize = 0;
        kd.cldnnStyle.inputBlockWidth = 0;

        kd.gemmStyle.globalWorkSizeDX = 1;
        kd.gemmStyle.globalWorkSizeDY = 1;
        kd.gemmStyle.globalWorkSizeDZ = 1;
        kd.gemmStyle.subBlockDimK = 1;
        kd.gemmStyle.subBlockDimM = 0;
        kd.gemmStyle.subBlockDimN = 0;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }

    KernelsData ConvolutionKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, const std::string exeMode, int autoTuneIndex) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<convolution_params>(params);
        convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

        if (NeedPaddedInput())
        {
            kd.reorderInput = CovolutionUpdateInputParams(newParams);
        }
        DispatchData runInfo = SetDefault(newParams, autoTuneIndex);
        
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
        FillCLKernelData(kernel, runInfo, params.engineInfo, finalKernelName, jit, entryPoint, exeMode, true, !newParams.bias.empty(), 1, newParams.int8_quantization, newParams.output_calibration);
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;
        kd.autoTuneIndex = autoTuneIndex;

        return{ kd };
    }
}
