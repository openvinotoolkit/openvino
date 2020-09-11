/*
// Copyright (c) 2019-2020 Intel Corporation
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
                                                          const DispatchData& kd) const {
    JitConstants jit = WeightBiasKernelBase::GetJitConstants(params);
    jit.Merge(GetFusedPrimitivesJitConstants(params, kd));

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

bool BinaryConvolutionKernelBase::CheckWorkGroups(const BinaryConvolutionKernelBase::DispatchData& kd) {
    if (kd.gws0 == 0 || kd.gws1 == 0 || kd.gws2 == 0 || kd.lws0 == 0 || kd.lws1 == 0 || kd.lws2 == 0) {
        return false;
    }

    if ((kd.gws0 % kd.lws0) != 0 || (kd.gws1 % kd.lws1) != 0 || (kd.gws2 % kd.lws2) != 0) {
        return false;
    }

    return true;
}

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernelBase::SetDefault(
    const binary_convolution_params& params,
    int) const {
    DispatchData kd;

    const auto& out = params.output;
    kd.fp16UnitUsed = out.GetDType() == Datatype::F16;
    std::vector<size_t> global;
    if (params.output.GetLayout() == DataLayout::bfyx || params.output.GetLayout() == DataLayout::byxf) {
        global = {out.X().v, out.Y().v, out.Feature().v * out.Batch().v};
    } else {
        global = {out.Feature().v * out.Batch().v, out.X().v, out.Y().v};
    }

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

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
    kd.efficiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    return kd;
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
    DispatchData runInfo = SetDefault(newParams, autoTuneIndex);

    if (!CheckWorkGroups(runInfo)) {
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
    auto cldnnJit = GetJitConstants(newParams, runInfo);
    auto entryPoint = GetEntryPoint(finalKernelName, newParams.layerID, options);
    auto jit = CreateJit(finalKernelName, cldnnJit, entryPoint);

    auto& kernel = kd.kernels[0];
    uint32_t fused_deps_total = 0;
    for (auto& fused_dep : newParams.fused_ops) {
        for (int i = 0; i < static_cast<int>(fused_dep.dep_size); i++) {
            kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, fused_deps_total});
            fused_deps_total++;
        }
    }

    FillCLKernelData(kernel,
                     runInfo,
                     params.engineInfo,
                     finalKernelName,
                     jit,
                     entryPoint,
                     exeMode,
                     true,
                     !newParams.bias.empty(),
                     1,
                     fused_deps_total);
    kernel.arguments.push_back({ArgumentDescriptor::Types::SPLIT, 0});

    kd.estimatedTime = runInfo.efficiency;
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

    const auto inputLimitX = (cp.output.X().v - 1) * cp.stride.x + (cp.filterSize.x - 1) * cp.dilation.x + 1;
    const auto inputLimitY = (cp.output.Y().v - 1) * cp.stride.y + (cp.filterSize.y - 1) * cp.dilation.y + 1;

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
