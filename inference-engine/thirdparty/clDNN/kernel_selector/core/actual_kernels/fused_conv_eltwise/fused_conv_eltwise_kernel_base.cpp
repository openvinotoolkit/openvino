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

#include "fused_conv_eltwise_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector 
{
    std::string fused_conv_eltwise_params::to_string() const
    {
        std::stringstream s;

        s << base_params::to_string() << "_";
        if (bias.empty())
        {
            s << "no_bias" << "_";
        }
        else
        {
            s << "bias_" << bias[0].PhysicalSize() << "_";
        }

        s << conv.filterSize.x << "_" << conv.filterSize.y << "_";
        s << conv.stride.x << "_" << conv.stride.y << "_";
        s << conv.dilation.x << "_" << conv.dilation.y << "_";
        s << conv.padding.x << "_" << conv.padding.y << "_";
        s << conv.split;

        return s.str();
    }

    ParamsKey fused_conv_eltwise_params::GetParamsKey() const
    {
        ParamsKey k = weight_bias_params::GetParamsKey();

        if (conv.split > 1)
        {
            k.EnableFusedConvEltwSplitSupport();
        }

        if (conv.dilation.x != 1 ||
            conv.dilation.y != 1)
        {
            k.EnableFusedConvEltwDilation();
        }

        if (conv.depthwise_separable_opt)
        {
            k.EnableFusedConvEltwDepthwiseSeparableOpt();
        }

        if (conv.transposed)
        {
            k.EnableFusedConvEltwTranspose();
        }

        if (conv.int8_quantization)
        {
            k.EnableFusedConvEltwInt8Quantization();
        }

        if (conv.output_calibration)
        {
            k.EnableFusedConvEltwOutputCalibration();
        }

        if (conv.local_convolution)
        {
            k.EnableFusedConvEltwLocalConvolution();
        }

        if (second_input_in_output)
        {
            k.EnableFusedConvEltwiseRWOutOpt();
        }

        return k;
    }

    bool fused_conv_eltwise_kernel_base::Validate(const Params& p, const optional_params& o) const
    {
        if (p.GetType() != KernelType::FUSED_CONV_ELTWISE ||
            o.GetType() != KernelType::FUSED_CONV_ELTWISE)
        {
            return false;
        }

        const fused_conv_eltwise_params& params = static_cast<const fused_conv_eltwise_params&>(p);
        const fused_conv_eltwise_optional_params& optParams = static_cast<const fused_conv_eltwise_optional_params&>(o);

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

    JitConstants fused_conv_eltwise_kernel_base::GetJitConstants(const fused_conv_eltwise_params& params, const DispatchData& kd) const
    {
        JitConstants mem_consts = WeightBiasKernelBase::GetJitConstants(params);
        const auto& padding = params.conv.padding;
        const auto& input = params.inputs[0];

        int64_t input_offset_with_padding = (int64_t)input.GetFirstElementOffset() - padding.x*input.X().pitch - input.Y().pitch*padding.y;
        input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

        mem_consts.AddConstants({
            MakeJitConstant("STRIDE",                       params.conv.stride),
            MakeJitConstant("PADDING",                      params.conv.padding),
            MakeJitConstant("DILATION",                     params.conv.dilation),
            MakeJitConstant("FILTER_ARRAY_NUM",             params.conv.split),
            MakeJitConstant("INPUT0_OFFSET_WITH_PADDING",   input_offset_with_padding),
            MakeJitConstant("DEPTHWISE_SEPARABLE_OPT",      params.conv.depthwise_separable_opt),
            MakeJitConstant("QUANTIZATION_TERM",            params.conv.int8_quantization),
        });

        if (params.conv.int8_quantization)
        {
            mem_consts.AddConstants({ MakeJitConstant("W_QF", params.conv.weights_quantization_factors[0]) });
            mem_consts.AddConstants({ MakeJitConstant("I_QF",params.conv.input_quantization_factor) });

            if (params.conv.output_calibration)
            {
                mem_consts.AddConstant(MakeJitConstant("CALIBRATION_TERM", params.conv.output_calibration));
                mem_consts.AddConstant(MakeJitConstant("O_QF", params.conv.output_calibration_factors[0]));

            }
            else
                mem_consts.AddConstants({ MakeJitConstant("O_QF", params.conv.output_quantization_factor) });
        }

        if (params.conv.local_convolution)
        {
            mem_consts.AddConstants({ MakeJitConstant("LOCAL_CONVOLUTION", params.conv.local_convolution) });
        }

        JitConstants eltw_activations = MakeActivationJitConstants(params.eltw.activation, "_ELTW");
        mem_consts.Merge(eltw_activations);

        mem_consts.AddConstant(MakeJitConstant("IN_OUT_OPT", params.second_input_in_output ? 1 : 0));

        std::vector<uint32_t> unrollLoopParams{
            params.conv.filterSize.x,
            params.conv.filterSize.y,
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

    bool fused_conv_eltwise_kernel_base::CheckWorkGroups(const fused_conv_eltwise_kernel_base::DispatchData& kd)
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

    bool fused_conv_eltwise_kernel_base::CheckPitchForSplitOnly(const fused_conv_eltwise_params& params)
    {
        // TODO: it's better to add pitch+offset support than handle this case
        return CheckTensorForSplit(params.inputs[0], params.conv.split);
    }

    fused_conv_eltwise_kernel_base::DispatchData fused_conv_eltwise_kernel_base::SetDefault(const fused_conv_eltwise_params& params, int) const
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

    KernelsData fused_conv_eltwise_kernel_base::GetCommonKernelsData(const Params& params, const optional_params& options, const std::string exeMode, int autoTuneIndex) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<fused_conv_eltwise_params>(params);
        fused_conv_eltwise_params& newParams = *static_cast<fused_conv_eltwise_params*>(kd.params.get());

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
        FillCLKernelData(kernel, runInfo, params.engineInfo, finalKernelName, jit, entryPoint, exeMode, true, !newParams.bias.empty(), 1, newParams.conv.int8_quantization, newParams.conv.output_calibration);
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });
        // eltwise's second input
        if(newParams.second_input_in_output)
        {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
        }
        else
        {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });
        }
        if (!newParams.eltw.output_calibration_factors.empty())
            kernel.arguments.push_back({ArgumentDescriptor::Types::OUTPUT_CALIBRATION_FACTORS, 1});

        kd.estimatedTime = runInfo.effiency;
        kd.autoTuneIndex = autoTuneIndex;

        return{ kd };
    }

    std::string fused_conv_eltwise_kernel_base::GetAutoTuneOptions(int autoTuneIndex) const
    {
        if ((autoTuneIndex >= 0) && (autoTuneIndex < (int)autoTuneOptions.size()))
        {
            return autoTuneOptions[autoTuneIndex];
        }

        return DEFAULT;
    }

    KernelsData fused_conv_eltwise_kernel_base::GetTunedKernelsDataByIndex(const Params& params, const optional_params& options, const int autoTuneIndex) const
    {
        return GetCommonKernelsData(params, options, GetAutoTuneOptions(autoTuneIndex), autoTuneIndex);
    }

    KernelsData fused_conv_eltwise_kernel_base::GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelsData res = {};

        for (size_t i = 0; i < autoTuneOptions.size(); i++)
        {
            KernelsData kd = GetTunedKernelsDataByIndex(params, options, (int)i);
            if (!kd.empty())
            {
                res.emplace_back(kd[0]);
            }
        }
        
        return res;
    }

    static DataTensor GetConvolutionBFYXPaddedTensor(const fused_conv_eltwise_params& cp)
    {
        assert(cp.inputs[0].GetDims().size() == 4U);

        DataTensor t = cp.inputs[0];
        std::vector<Tensor::Pad> pad{ { 0,0 },{ 0,0 },{ 0,0 },{ 0,0 } };

        auto& conv = cp.conv;

        pad[0].before = conv.padding.x;
        pad[1].before = conv.padding.y;


        const auto inputLimitX = (cp.output.X().v - 1) * conv.stride.x + (conv.filterSize.x - 1) * conv.dilation.x + 1;
        const auto inputLimitY = (cp.output.Y().v - 1) * conv.stride.y + (conv.filterSize.y - 1) * conv.dilation.y + 1;

        pad[0].after = (size_t)std::max((int)inputLimitX - (int)t.X().v - (int)pad[0].before, (int)0);
        pad[1].after = (size_t)std::max((int)inputLimitY - (int)t.Y().v - (int)pad[1].before, (int)0);

        Tensor::NDims dims(4);
        const Tensor::NDims& orgDims = cp.inputs[0].GetDims();
        size_t pitch = 1;
        for (size_t i = 0; i < dims.size(); i++)
        {
            dims[i].pad = pad[i];
            dims[i].v = orgDims[i].v;
            dims[i].pitch = pitch;
            pitch *= dims[i].LogicalDimPadded();
        }

        return{ dims, t.GetDType(), t.GetLayout() };
    }

    bool CheckConvolutionPaddedInputDesc(const fused_conv_eltwise_params& params, const DataTensor& reqDesc)
    {
        bool properPadding =
            reqDesc.X().pad.before <= params.inputs[0].X().pad.before &&
            reqDesc.Y().pad.before <= params.inputs[0].Y().pad.before &&
            reqDesc.Feature().pad.before <= params.inputs[0].Feature().pad.before &&
            reqDesc.Batch().pad.before <= params.inputs[0].Batch().pad.before;

        properPadding &=
            reqDesc.X().pad.after <= params.inputs[0].X().pad.after &&
            reqDesc.Y().pad.after <= params.inputs[0].Y().pad.after &&
            reqDesc.Feature().pad.after <= params.inputs[0].Feature().pad.after &&
            reqDesc.Batch().pad.after <= params.inputs[0].Batch().pad.after;

        properPadding &= ((params.conv.padding.x == 0 && params.conv.padding.y == 0) || params.inputs[0].GetPaddedVal() == 0.f);

        return properPadding;
    }

    bool CovolutionUpdateInputParams(fused_conv_eltwise_params& params)
    {
        const auto req_input = GetConvolutionBFYXPaddedTensor(params);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(params, req_input);

        if (!bProperInputDesc)
        {
            params.inputs[0] = req_input;
            return true;
        }

        return false;
    }

    bool FusedConvolutionEltwiseCheckInput(const Params& p, const optional_params& o)
    {
        const fused_conv_eltwise_params& params = static_cast<const fused_conv_eltwise_params&>(p);
        const fused_conv_eltwise_optional_params& optParams = static_cast<const fused_conv_eltwise_optional_params&>(o);

        const auto req_input = GetConvolutionBFYXPaddedTensor(params);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(params, req_input);
        const bool bInputPadded = optParams.allowInputReordering || bProperInputDesc;

        if (!bInputPadded)
        {
            return false;
        }

        return true;
    }

}
