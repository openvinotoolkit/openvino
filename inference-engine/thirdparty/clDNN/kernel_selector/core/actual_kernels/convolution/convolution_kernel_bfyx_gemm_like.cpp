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


#include "convolution_kernel_bfyx_gemm_like.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey ConvolutionKernel_bfyx_GEMMLike::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableSubGroup();
    // k.EnableSubGroupShort(); // we need it for FP16 only. we check it on the Validate phase
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    return k;
}

std::string ConvolutionKernel_bfyx_GEMMLike::GetKernelName(const convolution_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::F32) {
        return kernelName + "_fp32";
    } else {
        return kernelName + "_fp16";
    }
}

JitConstants ConvolutionKernel_bfyx_GEMMLike::GetJitConstants(const convolution_params& params,
                                                              const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstants({
        MakeJitConstant("ALIGNED_OFM_PER_GROUP", RoundUp(params.output.Feature().v / params.groups, dispatchData.gemmStyle.subBlockDimN)),
        MakeJitConstant("DX", dispatchData.gemmStyle.globalWorkSizeDX),
        MakeJitConstant("DY", dispatchData.gemmStyle.globalWorkSizeDY),
        MakeJitConstant("FILTER_SIZE_X_DIV2", params.filterSize.x / 2),
        MakeJitConstant("INPUT_BUFFER_WIDTH_PADDED", ""),  // TODO: enable non padding path again
        MakeJitConstant("INPUT_BUFFER_HEIGHT_PADDED", ""),
    });

    if (CeilDiv(RoundUp(params.output.X().v * params.output.Y().v, dispatchData.gemmStyle.subBlockDimM),
                dispatchData.gemmStyle.globalWorkSizeDY) %
            dispatchData.lws[1] !=
        0)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));

    return jit;
}

ConvolutionKernel_bfyx_GEMMLike::Parent::DispatchData ConvolutionKernel_bfyx_GEMMLike::SetDefault(
    const convolution_params& arg,
    int autoTuneIndex) const {
    DispatchData dispatchData = Parent::SetDefault(arg, autoTuneIndex);

    dispatchData.lws[0] = 1;
    dispatchData.lws[2] = 1;

    if (arg.inputs[0].GetDType() == Datatype::F16) {
        dispatchData.gemmStyle = {1, arg.filterSize.x, 32, 32, 1, 1};
        dispatchData.lws[1] = 16;
        dispatchData.efficiency = FORCE_PRIORITY_6;
    } else {
        dispatchData.gemmStyle = {2, arg.filterSize.x, 32, 32, 2, 1};
        dispatchData.lws[1] = 8;
        dispatchData.efficiency = FORCE_PRIORITY_8;
    }

    size_t sgemm_m = RoundUp(arg.output.X().v * arg.output.Y().v, dispatchData.gemmStyle.subBlockDimM);
    size_t sgemm_n = RoundUp(arg.output.Feature().v / arg.groups, dispatchData.gemmStyle.subBlockDimN);

    dispatchData.gws[0] = RoundUp(CeilDiv(sgemm_n, dispatchData.gemmStyle.globalWorkSizeDX), dispatchData.lws[0]);
    dispatchData.gws[1] = RoundUp(CeilDiv(sgemm_m, dispatchData.gemmStyle.globalWorkSizeDY), dispatchData.lws[1]);
    dispatchData.gws[2] = arg.output.Batch().v * arg.groups;

    return dispatchData;
}

bool ConvolutionKernel_bfyx_GEMMLike::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    if (!params.engineInfo.bSubGroupShortSupport && params.inputs[0].GetDType() == Datatype::F16) {
        return false;
    }

    return true;
}

WeightsLayout ConvolutionKernel_bfyx_GEMMLike::GetPreferredWeightsLayout(
        const convolution_params &params) const {
    if (params.inputs[0].GetDType() == Datatype::F16) {
        return (params.groups > 1) ? WeightsLayout::giy_xs_os_xsv2_osv16__ao32 : WeightsLayout::iy_xs_os_xsv2_osv16__ao32;
    } else {
        return (params.groups > 1) ? WeightsLayout::giy_xs_os_xsv2_osv8__ao32 : WeightsLayout::iy_xs_os_xsv2_osv8__ao32;
    }
}

KernelsData ConvolutionKernel_bfyx_GEMMLike::GetKernelsData(const Params& params,
                                                            const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector