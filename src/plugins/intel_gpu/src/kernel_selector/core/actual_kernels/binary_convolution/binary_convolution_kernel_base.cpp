// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "binary_convolution_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <vector>
#include <algorithm>

namespace kernel_selector {
bool BinaryConvolutionKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::BINARY_CONVOLUTION || o.GetType() != KernelType::BINARY_CONVOLUTION) {
        return false;
    }

    const binary_convolution_params& params = static_cast<const binary_convolution_params&>(p);
    const binary_convolution_optional_params& optParams = static_cast<const binary_convolution_optional_params&>(o);

    bool bSupportedWeightsLayout = params.weights.GetLayout() == GetPreferredWeightLayout(params);

    const bool bWeightsOK = bSupportedWeightsLayout || optParams.allowStaticInputReordering;

    if (!bWeightsOK) {
        return false;
    }

    return true;
}

JitConstants BinaryConvolutionKernelBase::GetJitConstants(const binary_convolution_params& params,
                                                          const DispatchData& dispatchData) const {
    JitConstants jit = WeightBiasKernelBase::GetJitConstants(params);
    jit.Merge(GetFusedPrimitivesJitConstants(params, dispatchData));

    jit.AddConstants({
        MakeJitConstant("STRIDE", params.stride),
        MakeJitConstant("PADDING", params.padding),
        MakeJitConstant("DILATION", params.dilation),
    });

    jit.Merge(MakeTypeJitConstants(params.out_dt, "CONV_RESULT"));

    return jit;
}

JitConstants BinaryConvolutionKernelBase::GetFusedPrimitivesJitConstants(const binary_convolution_params& /*params*/,
                                                                         const DispatchData& /*kd*/) const {
    return {};
}

bool BinaryConvolutionKernelBase::CheckWorkGroups(const BinaryConvolutionKernelBase::DispatchData& dispatchData) {
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

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernelBase::SetDefault(const binary_convolution_params& params,
                                                                                  int) const {
    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& out = params.outputs[0];
    std::vector<size_t> global;
    if (out_layout == DataLayout::bfyx || out_layout == DataLayout::byxf) {
        global = {out.X().v, out.Y().v, out.Feature().v * out.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    } else {
        global = {out.Feature().v * out.Batch().v, out.X().v, out.Y().v};
        dims_by_gws = {{Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH},
                       {Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y}};
    }

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo, in_layout, out_layout, dims_by_gws);

    dispatchData.gws = global;
    dispatchData.lws = local;

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

KernelsData BinaryConvolutionKernelBase::GetCommonKernelsData(const Params& params,
                                                              const optional_params& options,
                                                              const std::string exeMode,
                                                              int autoTuneIndex) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<binary_convolution_params>(params);
    binary_convolution_params& newParams = *static_cast<binary_convolution_params*>(kd.params.get());

    if (NeedPaddedInput()) {
        kd.reorderInput = CovolutionBinaryUpdateInputParams(newParams);
    }
    DispatchData dispatchData = SetDefault(newParams, autoTuneIndex);

    if (!CheckWorkGroups(dispatchData)) {
        // Internal Error - wrong calculation of global/local work group sizes
        return {};
    }

    bool succeed = UpdateWeightsParams(newParams,
                                       options,
                                       GetPreferredWeightLayout(newParams),
                                       kd.weightsReorderParams,
                                       GetSupportedKey());

    if (!succeed) {
        return {};
    }

    auto finalKernelName = GetKernelName(newParams);
    auto cldnnJit = GetJitConstants(newParams, dispatchData);
    auto entryPoint = GetEntryPoint(finalKernelName, newParams.layerID, params, options);
    auto jit = CreateJit(finalKernelName, cldnnJit, entryPoint);

    auto& kernel = kd.kernels[0];
    uint32_t fused_deps_total = 0;
    for (auto& fused_dep : newParams.fused_ops) {
        for (int i = 0; i < static_cast<int>(fused_dep.dep_size); i++) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, fused_deps_total});
            fused_deps_total++;
        }
    }

    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     finalKernelName,
                     jit,
                     entryPoint,
                     exeMode,
                     true,
                     !newParams.bias.empty(),
                     1,
                     fused_deps_total);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SPLIT, 0});

    kd.autoTuneIndex = autoTuneIndex;

    return {kd};
}

bool CheckConvolutionBinaryPaddedInputDesc(const binary_convolution_params& params, const DataTensor& reqDesc) {
    assert(params.inputs.size() == 1);

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

static DataTensor GetConvolutionBFYXPaddedTensor(const binary_convolution_params& cp) {
    assert(cp.inputs.size() == 1);
    assert(cp.inputs[0].GetDims().size() == 4U);

    DataTensor t = cp.inputs[0];
    std::vector<Tensor::Pad> pad{{0, 0}, {0, 0}, {0, 0}, {0, 0}};

    pad[0].before = cp.padding.x;
    pad[1].before = cp.padding.y;

    const auto inputLimitX = (cp.outputs[0].X().v - 1) * cp.stride.x + (cp.filterSize.x - 1) * cp.dilation.x + 1;
    const auto inputLimitY = (cp.outputs[0].Y().v - 1) * cp.stride.y + (cp.filterSize.y - 1) * cp.dilation.y + 1;

    pad[0].after = (size_t)std::max(static_cast<int>(inputLimitX) - static_cast<int>(t.X().v) - static_cast<int>(pad[0].before), static_cast<int>(0));
    pad[1].after = (size_t)std::max(static_cast<int>(inputLimitY) - static_cast<int>(t.Y().v) - static_cast<int>(pad[1].before), static_cast<int>(0));

    Tensor::NDims dims(4);
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

bool CovolutionBinaryCheckInput(const Params& p, const optional_params& o) {
    const binary_convolution_params& params = static_cast<const binary_convolution_params&>(p);
    const binary_convolution_optional_params& optParams = static_cast<const binary_convolution_optional_params&>(o);

    const auto req_input = GetConvolutionBFYXPaddedTensor(params);
    const bool bProperInputDesc = CheckConvolutionBinaryPaddedInputDesc(params, req_input);
    const bool bInputPadded = optParams.allowInputReordering || bProperInputDesc;

    if (!bInputPadded) {
        return false;
    }

    return true;
}

bool CovolutionBinaryUpdateInputParams(binary_convolution_params& params) {
    const auto req_input = GetConvolutionBFYXPaddedTensor(params);
    const bool bProperInputDesc = CheckConvolutionBinaryPaddedInputDesc(params, req_input);

    if (!bProperInputDesc) {
        params.inputs[0] = req_input;
        return true;
    }

    return false;
}

std::string BinaryConvolutionKernelBase::GetAutoTuneOptions(int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    return DEFAULT;
}

KernelsData BinaryConvolutionKernelBase::GetTunedKernelsDataByIndex(const Params& params,
                                                                    const optional_params& options,
                                                                    const int autoTuneIndex) const {
    return GetCommonKernelsData(params, options, GetAutoTuneOptions(autoTuneIndex), autoTuneIndex);
}

KernelsData BinaryConvolutionKernelBase::GetKernelsDataForAutoTune(const Params& params,
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
}  // namespace kernel_selector
