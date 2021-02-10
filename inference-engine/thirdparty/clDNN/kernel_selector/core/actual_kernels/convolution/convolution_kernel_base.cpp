/*
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
*/

#include "convolution_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <vector>
#include <algorithm>

namespace kernel_selector {
bool ConvolutionKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::CONVOLUTION || o.GetType() != KernelType::CONVOLUTION) {
        return false;
    }

    const convolution_params& params = static_cast<const convolution_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

JitConstants ConvolutionKernelBase::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    JitConstants mem_consts = WeightBiasKernelBase::GetJitConstants(params);
    mem_consts.Merge(GetFusedPrimitivesJitConstants(params, dispatchData));
    const auto& padding = params.padding;
    const auto& input = params.inputs[0];

    int64_t input_offset_with_padding =
        (int64_t)input.GetFirstElementOffset() - padding.x * input.X().pitch - input.Y().pitch * padding.y;
    input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

    mem_consts.AddConstants({
        MakeJitConstant("STRIDE", params.stride),
        MakeJitConstant("PADDING", params.padding),
        MakeJitConstant("DILATION", params.dilation),
        MakeJitConstant("FILTER_ARRAY_NUM", params.split * params.groups),
        MakeJitConstant("INPUT0_OFFSET_WITH_PADDING", input_offset_with_padding),
        MakeJitConstant("DEPTHWISE_SEPARABLE_OPT", params.depthwise_separable_opt),
        MakeJitConstant("GROUPED", (params.groups > 1) ? 1 : 0),
    });

    if (params.quantization != QuantizationType::NONE) {
        mem_consts.AddConstants({MakeJitConstant("QUANTIZATION_TERM", 1)});
    }

    if (params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS || params.quantization == QuantizationType::ASYMMETRIC_DATA) {
        mem_consts.AddConstants({MakeJitConstant("ASYMMETRIC_DATA_QUANTIZATION", 1)});
        if (!params.activations_zero_points.empty()) {
            mem_consts.AddConstants({MakeJitConstant("ACTIVATIONS_ZERO_POINTS", params.activations_zero_points[0])});
        }
        if (params.HasCompensation()) {
            mem_consts.AddConstants({MakeJitConstant("COMPENSATION_TERM", 1)});
            mem_consts.AddConstants({MakeJitConstant("COMPENSATION", params.compensation[0])});
        }
    }
    if (params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS || params.quantization == QuantizationType::ASYMMETRIC_WEIGHTS) {
        mem_consts.AddConstants({MakeJitConstant("ASYMMETRIC_WEIGHTS_QUANTIZATION", 1)});
        if (!params.weights_zero_points.empty()) {
            mem_consts.AddConstants({MakeJitConstant("WEIGHTS_ZERO_POINTS", params.weights_zero_points[0])});
        }
    }
    if (params.quantization == QuantizationType::SYMMETRIC) {
        mem_consts.AddConstants({MakeJitConstant("SYMMETRIC_QUANTIZATION", 1)});
    }

    if (params.local_convolution) {
        mem_consts.AddConstants({MakeJitConstant("LOCAL_CONVOLUTION", params.local_convolution)});
    }

    if (params.deformable_mode) {
        mem_consts.AddConstants({MakeJitConstant("DEFORMABLE_GROUPS", params.deformable_groups)});
        mem_consts.AddConstants({MakeJitConstant("DEFORMABLE_MODE", params.deformable_mode)});
    }

    std::vector<uint32_t> unrollLoopParams{params.filterSize.x,
                                           params.filterSize.y,
                                           (uint32_t)dispatchData.gemmStyle.globalWorkSizeDX,
                                           (uint32_t)dispatchData.gemmStyle.globalWorkSizeDY,
                                           (uint32_t)dispatchData.gemmStyle.globalWorkSizeDZ,
                                           (uint32_t)dispatchData.gemmStyle.subBlockDimM,
                                           (uint32_t)dispatchData.gemmStyle.subBlockDimK,
                                           (uint32_t)dispatchData.gemmStyle.subBlockDimN};

    auto loopCount = *std::max_element(unrollLoopParams.begin(), unrollLoopParams.end());

    JitConstants mem_consts_loop = MakeLoopUnrollParamsJitConstants(loopCount);
    mem_consts.Merge(mem_consts_loop);

    return mem_consts;
}

bool ConvolutionKernelBase::CheckWorkGroups(const ConvolutionKernelBase::DispatchData& dispatchData) {
    if (dispatchData.gws.size() != 3 || dispatchData.lws.size() != 3)
        return false;

    for (size_t i = 0; i < dispatchData.gws.size(); i++) {
        if (dispatchData.gws[i] == 0 || dispatchData.lws[i] == 0)
            return false;
        if ((dispatchData.gws[i] % dispatchData.lws[i]) != 0)
            return false;
    }

    return true;
}

namespace {
bool CheckTensorForSplit(const DataTensor& t, uint32_t split) {
    if (t.PitchesDifferFromLogicalDims()) {
        auto feature = t.Feature();
        auto featureIndex = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::FEATURE);
        if (featureIndex >= 0 && featureIndex + 1 < static_cast<int>(DataTensor::ChannelsCount(t.GetLayout()))) {
            if (feature.v * split <= t.GetDims()[featureIndex + 1].pitch) {
                Tensor::NDims newDims = t.GetDims();
                newDims[featureIndex].v = feature.v * split;

                DataTensor newTensor{newDims,
                                     t.GetDType(),
                                     t.GetLayout(),
                                     t.GetViewOffset(),
                                     t.PhysicalSize(),
                                     t.GetPaddedVal()};

                if (newTensor.PitchesDifferFromLogicalDims() == false) {
                    return true;
                }
            }
        }

        return false;
    }

    return true;
}
}  // namespace

bool ConvolutionKernelBase::CheckPitchForSplitOnly(const convolution_params& params) {
    // TODO: it's better to add pitch+offset support than handle this case
    return CheckTensorForSplit(params.inputs[0], params.split);
}

ConvolutionKernelBase::DispatchData ConvolutionKernelBase::SetDefault(const convolution_params& params, int) const {
    DispatchData dispatchData;

    const auto& out = params.output;
    if (params.output.GetLayout() == DataLayout::bfyx || params.output.GetLayout() == DataLayout::byxf) {
        dispatchData.gws = {out.X().v, out.Y().v, out.Feature().v * out.Batch().v};
    } else if (params.output.GetLayout() == DataLayout::bfzyx) {
        dispatchData.gws = {out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v};
    } else {
        dispatchData.gws = {out.Feature().v * out.Batch().v, out.X().v, out.Y().v};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    dispatchData.cldnnStyle.blockWidth = 1;
    dispatchData.cldnnStyle.blockHeight = 1;
    dispatchData.cldnnStyle.prefetch = 0;
    dispatchData.cldnnStyle.inputBlockArraySize = 0;
    dispatchData.cldnnStyle.inputBlockWidth = 0;

    dispatchData.gemmStyle.globalWorkSizeDX = 1;
    dispatchData.gemmStyle.globalWorkSizeDY = 1;
    dispatchData.gemmStyle.globalWorkSizeDZ = 1;
    dispatchData.gemmStyle.subBlockDimK = 1;
    dispatchData.gemmStyle.subBlockDimM = 0;
    dispatchData.gemmStyle.subBlockDimN = 0;
    return dispatchData;
}

KernelsData ConvolutionKernelBase::GetCommonKernelsData(const Params& params,
                                                        const optional_params& options,
                                                        const std::string exeMode,
                                                        int autoTuneIndex) const {
    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    bool succeed = UpdateWeightsParams(newParams,
                                       options,
                                       GetPreferredWeightsLayout(newParams),
                                       kd.weightsReorderParams,
                                       GetSupportedKey(),
                                       newParams.groups,
                                       newParams.transposed);

    bool bSupportedWeightsLayout = newParams.weights.GetLayout() == GetPreferredWeightsLayout(newParams);
    const bool bWeightsOK = bSupportedWeightsLayout || options.allowStaticInputReordering;

    if (!succeed || !bWeightsOK) {
        return {};
    }

    if (NeedPaddedInput()) {
        kd.reorderInput = CovolutionUpdateInputParams(newParams);

        if (kd.reorderInput && !options.allowInputReordering)
            return {};
    }
    DispatchData dispatchData = SetDefault(newParams, autoTuneIndex);

    if (!CheckWorkGroups(dispatchData)) {
        // Internal Error - wrong calculation of global/local work group sizes
        return {};
    }

    auto finalKernelName = GetKernelName(newParams);
    auto cldnnJit = GetJitConstants(newParams, dispatchData);
    auto entryPoint = GetEntryPoint(finalKernelName, newParams.layerID, options);
    auto jit = CreateJit(finalKernelName, cldnnJit, entryPoint);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     finalKernelName,
                     jit,
                     entryPoint,
                     exeMode,
                     true,
                     !newParams.bias.empty(),
                     1);

    if (newParams.deformable_mode) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    }

    if (!newParams.weights_zero_points.empty())
        kernel.arguments.push_back({ArgumentDescriptor::Types::WEIGHTS_ZERO_POINTS, 1});
    if (!newParams.activations_zero_points.empty())
        kernel.arguments.push_back({ArgumentDescriptor::Types::ACTIVATIONS_ZERO_POINTS, 1});
    if (!newParams.compensation.empty())
        kernel.arguments.push_back({ArgumentDescriptor::Types::COMPENSATION, 1});

    uint32_t fused_deps_total = 0;
    for (auto& fused_dep : newParams.fused_ops) {
        for (int i = 0; i < static_cast<int>(fused_dep.dep_size); i++) {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, fused_deps_total });
            fused_deps_total++;
        }
    }
    kernel.arguments.push_back({ArgumentDescriptor::Types::SPLIT, 0});

    kd.autoTuneIndex = autoTuneIndex;

    return {kd};
}

bool CheckConvolutionPaddedInputDesc(const convolution_params& params, const DataTensor& reqDesc) {
    assert(params.inputs.size() >= 1);

    bool properPadding = reqDesc.X().pad.before <= params.inputs[0].X().pad.before &&
                         reqDesc.Y().pad.before <= params.inputs[0].Y().pad.before &&
                         reqDesc.Feature().pad.before <= params.inputs[0].Feature().pad.before &&
                         reqDesc.Batch().pad.before <= params.inputs[0].Batch().pad.before;

    properPadding &= reqDesc.X().pad.after <= params.inputs[0].X().pad.after &&
                     reqDesc.Y().pad.after <= params.inputs[0].Y().pad.after &&
                     reqDesc.Feature().pad.after <= params.inputs[0].Feature().pad.after &&
                     reqDesc.Batch().pad.after <= params.inputs[0].Batch().pad.after;

    properPadding &= ((params.padding.x == 0 && params.padding.y == 0) || params.inputs[0].GetPaddedVal() == 0.f);

    return properPadding;
}

static DataTensor GetConvolutionBFYXPaddedTensor(const convolution_params& cp) {
    assert(cp.inputs.size() >= 1);
    auto ndims = cp.inputs[0].GetDims().size();

    DataTensor t = cp.inputs[0];
    std::vector<Tensor::Pad> pad{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} };

    pad[0].before = cp.padding.x;
    pad[1].before = cp.padding.y;
    pad[2].before = cp.padding.z;


    const auto inputLimitX = (cp.output.X().v - 1) * cp.stride.x + (cp.filterSize.x - 1) * cp.dilation.x + 1;
    const auto inputLimitY = (cp.output.Y().v - 1) * cp.stride.y + (cp.filterSize.y - 1) * cp.dilation.y + 1;
    const auto inputLimitZ = (cp.output.Z().v - 1) * cp.stride.z + (cp.filterSize.z - 1) * cp.dilation.z + 1;


    pad[0].after = (size_t)std::max(static_cast<int>(inputLimitX) - static_cast<int>(t.X().v) - static_cast<int>(pad[0].before), static_cast<int>(0));
    pad[1].after = (size_t)std::max(static_cast<int>(inputLimitY) - static_cast<int>(t.Y().v) - static_cast<int>(pad[1].before), static_cast<int>(0));
    pad[2].after = (size_t)std::max(static_cast<int>(inputLimitZ) - static_cast<int>(t.Z().v) - static_cast<int>(pad[2].before), static_cast<int>(0));

    Tensor::NDims dims(ndims);
    const Tensor::NDims& orgDims = cp.inputs[0].GetDims();
    size_t pitch = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        dims[i].pad = pad[i];
        dims[i].v = orgDims[i].v;
        dims[i].pitch = pitch;
        pitch *= dims[i].LogicalDimPadded();
    }

    return {dims, t.GetDType(), t.GetLayout()};
}

bool CovolutionCheckInput(const Params& p, const optional_params& o) {
    const convolution_params& params = static_cast<const convolution_params&>(p);
    const convolution_optional_params& optParams = static_cast<const convolution_optional_params&>(o);

    const auto req_input = GetConvolutionBFYXPaddedTensor(params);
    const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(params, req_input);
    const bool bInputPadded = optParams.allowInputReordering || bProperInputDesc;

    if (!bInputPadded) {
        return false;
    }

    return true;
}

bool CovolutionUpdateInputParams(convolution_params& params) {
    const auto req_input = GetConvolutionBFYXPaddedTensor(params);
    const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(params, req_input);

    if (!bProperInputDesc) {
        params.inputs[0] = req_input;
        return true;
    }

    return false;
}

std::string ConvolutionKernelBase::GetAutoTuneOptions(int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    return DEFAULT;
}

KernelsData ConvolutionKernelBase::GetTunedKernelsDataByIndex(const Params& params,
                                                              const optional_params& options,
                                                              const int autoTuneIndex) const {
    return GetCommonKernelsData(params, options, GetAutoTuneOptions(autoTuneIndex), autoTuneIndex);
}

KernelsData ConvolutionKernelBase::GetKernelsDataForAutoTune(const Params& params,
                                                             const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

JitConstants ConvolutionKernelBase::GetFusedPrimitivesJitConstants(const convolution_params& /*params*/,
                                                                   const DispatchData& /*kd*/) const {
    return {};
}


Datatype ConvolutionKernelBase::GetPackedType(Datatype dt, size_t pack_size) const {
    if (dt == Datatype::UINT8) {
        return pack_size == 4 ? Datatype::UINT32 : pack_size == 2 ? Datatype::UINT16 : dt;
    } else if (dt == Datatype::INT8) {
        return pack_size == 4 ?  Datatype::INT32 : pack_size == 2 ? Datatype::INT16 : dt;
    } else {
        return dt;
    }
}

Datatype ConvolutionKernelBase::GetPackedInputType(const convolution_params& params) const {
    return GetPackedType(params.inputs[0].GetDType());
}

Datatype ConvolutionKernelBase::GetPackedOutputType(const convolution_params& params) const {
    return GetPackedType(params.output.GetDType());
}

Datatype ConvolutionKernelBase::GetActivationType(const convolution_params& params) const {
    bool quantized_weights = false;
    bool quantized_inputs = false;

    if (params.inputs[0].GetDType() == Datatype::UINT8 ||
        params.inputs[0].GetDType() == Datatype::INT8)
        quantized_inputs = true;

    if (params.weights.GetDType() == WeightsType::UINT8 ||
        params.weights.GetDType() == WeightsType::INT8)
        quantized_weights = true;

    if (params.quantization != QuantizationType::NONE || quantized_inputs || quantized_weights)
        return Datatype::F32;

    if (params.output.GetDType() == Datatype::UINT8 ||
        params.output.GetDType() == Datatype::INT8) {
        if (params.inputs[0].GetDType() == Datatype::F32) {
            return Datatype::F32;
        } else if (params.inputs[0].GetDType() == Datatype::F16) {
            return Datatype::F16;
        }
    }

    return GetUnitType(params);
}

Datatype ConvolutionKernelBase::GetAccumulatorType(const convolution_params& params) const {
    if (params.quantization != QuantizationType::NONE)
        return Datatype::INT32;

    bool quantized_weights = false;
    bool quantized_inputs = false;

    if (params.inputs[0].GetDType() == Datatype::UINT8 ||
        params.inputs[0].GetDType() == Datatype::INT8)
        quantized_inputs = true;

    if (params.weights.GetDType() == WeightsType::UINT8 ||
        params.weights.GetDType() == WeightsType::INT8)
        quantized_weights = true;

    // This case should be always false, because quantization type is not NONE
    if (quantized_inputs && quantized_weights)
        return Datatype::INT32;

    // If we either weights or input is quantized, then we use fp32 accumulator to avoid fp16 overflow
    if (quantized_inputs || quantized_weights)
        return Datatype::F32;

    return params.inputs[0].GetDType();
}

}  // namespace kernel_selector
