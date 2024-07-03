// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <vector>
#include <algorithm>

namespace kernel_selector {
bool ConvolutionKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::CONVOLUTION) {
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
    const auto& padding = params.padding_begin;
    const auto& input = params.inputs[0];

    int64_t input_offset_with_padding =
        (int64_t)input.GetFirstElementOffset() - padding.x * input.X().pitch - input.Y().pitch * padding.y;
    input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

    mem_consts.AddConstants({
        MakeJitConstant("STRIDE", params.stride),
        MakeJitConstant("PADDING", padding),
        MakeJitConstant("DILATION", params.dilation),
        MakeJitConstant("FILTER_ARRAY_NUM", params.groups),
        MakeJitConstant("INPUT0_OFFSET_WITH_PADDING", input_offset_with_padding),
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

    if (params.deformable_mode) {
        mem_consts.AddConstants({MakeJitConstant("DEFORMABLE_GROUPS", params.deformable_groups)});
        mem_consts.AddConstants({MakeJitConstant("DEFORMABLE_MODE", params.deformable_mode)});
        if (params.deformable_mask_enabled)
            mem_consts.AddConstants({MakeJitConstant("DEFORMABLE_MASK_ENABLED", params.deformable_mask_enabled)});
        if (params.bilinear_interpolation_pad)
            mem_consts.AddConstants({MakeJitConstant("BILINEAR_INTERPOLATION_PAD", params.bilinear_interpolation_pad)});
    }

    if (!params.is_shape_agnostic) {
        if (params.outputs[0].Batch().v == 1) {
            mem_consts.AddConstant(MakeJitConstant("SKIP_BATCH", 1));
        }
    }

    return mem_consts;
}

JitConstants ConvolutionKernelBase::GetJitConstantsWithLoopUnroll(const convolution_params& params, const DispatchData& dispatchData) const {
    JitConstants mem_consts = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

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

ConvolutionKernelBase::DispatchData ConvolutionKernelBase::SetDefault(const convolution_params& params, int) const {
    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& out = params.outputs[0];
    if (out_layout == DataLayout::bfyx || out_layout == DataLayout::byxf) {
        dispatchData.gws = {out.X().v, out.Y().v, out.Feature().v * out.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    } else if (out_layout == DataLayout::bfzyx) {
        dispatchData.gws = {out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    } else {
        dispatchData.gws = {out.Feature().v * out.Batch().v, out.X().v, out.Y().v};
        dims_by_gws = {{Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH},
                       {Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y}};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

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

void ConvolutionKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const convolution_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(prim_params.inputs[0].PhysicalSizeInBytes());
        kd.internalBufferDataType = prim_params.inputs[0].GetDType();
    };
}

KernelsData ConvolutionKernelBase::GetCommonKernelsData(const Params& params,
                                                        const std::string exeMode,
                                                        int autoTuneIndex) const {
    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto preferredWeightsLayout = GetPreferredWeightsLayout(newParams);
    bool succeed = UpdateWeightsParams(newParams,
                                       preferredWeightsLayout,
                                       kd.weightsReorderParams,
                                       GetSupportedKey(),
                                       newParams.groups,
                                       newParams.transposed);

    bool bSupportedWeightsLayout = newParams.weights.GetLayout() == preferredWeightsLayout;
    const bool bWeightsOK = bSupportedWeightsLayout || newParams.allowStaticInputReordering;

    if (!succeed || !bWeightsOK) {
        return {};
    }

    if (NeedPaddedInput()) {
        if (newParams.has_dynamic_inputs()) {
            if (!CheckConvolutionExplicitPaddings(newParams))
                return {};
        } else {
            kd.reorderInput = ConvolutionUpdateInputParams(newParams);

            if (kd.reorderInput && !newParams.allowInputReordering)
                return {};
        }
    }

    DispatchData dispatchData = SetDefault(newParams, autoTuneIndex);

    if (!params.is_shape_agnostic && !CheckWorkGroups(dispatchData)) {
        // Internal Error - wrong calculation of global/local work group sizes
        return {};
    }

    auto finalKernelName = GetKernelName(newParams);
    auto cldnnJit = GetJitConstants(newParams, dispatchData);
    auto entryPoint = GetEntryPoint(finalKernelName, newParams.layerID, params);
    auto jit = CreateJit(finalKernelName, cldnnJit, entryPoint);

    GetUpdateDispatchDataFunc(kd);

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
                     1, 0, 1,
                     newParams.is_shape_agnostic);

    if (newParams.deformable_mode) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        if (newParams.deformable_mask_enabled)
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
    }

    if (!newParams.weights_zero_points.empty())
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::WEIGHTS_ZERO_POINTS, 1});
    if (!newParams.activations_zero_points.empty())
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::ACTIVATIONS_ZERO_POINTS, 1});
    if (!newParams.compensation.empty())
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::COMPENSATION, 1});

    uint32_t fused_deps_total = 0;
    for (auto& fused_dep : newParams.fused_ops) {
        for (int i = 0; i < static_cast<int>(fused_dep.dep_size); i++) {
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, fused_deps_total });
            fused_deps_total++;
        }
    }
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

    properPadding &= ((params.padding_begin.x == 0 && params.padding_begin.y == 0) || params.inputs[0].GetPaddedVal() == 0.f);

    return properPadding;
}

static DataTensor GetConvolutionBFYXPaddedTensor(const convolution_params& cp) {
    assert(cp.inputs.size() >= 1);
    auto ndims = cp.inputs[0].GetDims().size();

    DataTensor t = cp.inputs[0];
    std::vector<Tensor::Pad> pad{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} };

    pad[0].before = cp.padding_begin.x;
    pad[1].before = cp.padding_begin.y;
    pad[2].before = cp.padding_begin.z;

    const auto inputLimitX = (cp.outputs[0].X().v - 1) * cp.stride.x + (cp.filterSize.x - 1) * cp.dilation.x + 1;
    const auto inputLimitY = (cp.outputs[0].Y().v - 1) * cp.stride.y + (cp.filterSize.y - 1) * cp.dilation.y + 1;
    const auto inputLimitZ = (cp.outputs[0].Z().v - 1) * cp.stride.z + (cp.filterSize.z - 1) * cp.dilation.z + 1;

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

bool CheckConvolutionExplicitPaddings(const convolution_params& conv_params) {
    if (!conv_params.has_explicit_paddings)
        return false;

    bool proper_padding = true;
    proper_padding &= conv_params.padding_begin.x == conv_params.inputs[0].X().pad.before &&
                      conv_params.padding_begin.y == conv_params.inputs[0].Y().pad.before &&
                      conv_params.padding_begin.z == conv_params.inputs[0].Z().pad.before;

    proper_padding &= conv_params.padding_end.x == conv_params.inputs[0].X().pad.after &&
                      conv_params.padding_end.y == conv_params.inputs[0].Y().pad.after &&
                      conv_params.padding_end.z == conv_params.inputs[0].Z().pad.after;

    return proper_padding;
}

bool ConvolutionCheckInput(const Params& p) {
    const convolution_params& params = static_cast<const convolution_params&>(p);

    if (params.has_dynamic_inputs())
        return CheckConvolutionExplicitPaddings(params);

    const auto req_input = GetConvolutionBFYXPaddedTensor(params);
    const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(params, req_input);
    const bool bInputPadded = params.allowInputReordering || bProperInputDesc;

    if (!bInputPadded) {
        return false;
    }

    return true;
}

bool ConvolutionUpdateInputParams(convolution_params& params) {
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

    return EXE_MODE_DEFAULT;
}

KernelsData ConvolutionKernelBase::GetTunedKernelsDataByIndex(const Params& params,
                                                              const int autoTuneIndex) const {
    return GetCommonKernelsData(params, GetAutoTuneOptions(autoTuneIndex), autoTuneIndex);
}

KernelsData ConvolutionKernelBase::GetKernelsDataForAutoTune(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, static_cast<int>(i));
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
    return GetPackedType(params.outputs[0].GetDType());
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

    if (params.outputs[0].GetDType() == Datatype::UINT8 ||
        params.outputs[0].GetDType() == Datatype::INT8) {
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
