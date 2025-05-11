// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_bfyx_GEMMLike::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_broadcast();

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
    JitConstants jit = Parent::GetJitConstantsWithLoopUnroll(params, dispatchData);

    jit.AddConstants({
        MakeJitConstant("ALIGNED_OFM_PER_GROUP", RoundUp(params.outputs[0].Feature().v / params.groups, dispatchData.gemmStyle.subBlockDimN)),
        MakeJitConstant("DX", dispatchData.gemmStyle.globalWorkSizeDX),
        MakeJitConstant("DY", dispatchData.gemmStyle.globalWorkSizeDY),
        MakeJitConstant("FILTER_SIZE_X_DIV2", params.filterSize.x / 2),
        MakeJitConstant("INPUT_BUFFER_WIDTH_PADDED", ""),  // TODO: enable non padding path again
        MakeJitConstant("INPUT_BUFFER_HEIGHT_PADDED", ""),
    });

    if (CeilDiv(RoundUp(params.outputs[0].X().v * params.outputs[0].Y().v, dispatchData.gemmStyle.subBlockDimM),
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
    } else {
        dispatchData.gemmStyle = {2, arg.filterSize.x, 32, 32, 2, 1};
        dispatchData.lws[1] = 8;
    }

    size_t sgemm_m = RoundUp(arg.outputs[0].X().v * arg.outputs[0].Y().v, dispatchData.gemmStyle.subBlockDimM);
    size_t sgemm_n = RoundUp(arg.outputs[0].Feature().v / arg.groups, dispatchData.gemmStyle.subBlockDimN);

    dispatchData.gws[0] = RoundUp(CeilDiv(sgemm_n, dispatchData.gemmStyle.globalWorkSizeDX), dispatchData.lws[0]);
    dispatchData.gws[1] = RoundUp(CeilDiv(sgemm_m, dispatchData.gemmStyle.globalWorkSizeDY), dispatchData.lws[1]);
    dispatchData.gws[2] = arg.outputs[0].Batch().v * arg.groups;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_bfyx_GEMMLike::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);

    return p.outputs[0].GetDType() == Datatype::F16 ? FORCE_PRIORITY_6 : FORCE_PRIORITY_8;
}

bool ConvolutionKernel_bfyx_GEMMLike::Validate(const Params& p) const {
    if (!Parent::Validate(p) || !ConvolutionCheckInput(p)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    if (!IsSIMDSizeSupported(p.engineInfo, 8) && params.inputs[0].GetDType() == Datatype::F32)
        return false;

    if (!params.engineInfo.supports_intel_subgroups_short && params.inputs[0].GetDType() == Datatype::F16) {
        return false;
    }

    // Limit filter_x_size to 32 because convolution ref kernel is faster than GEMMLike kernel when filter size is bigger.
    // 32 is chosen from filter size of customer model. May need to more measurement to pick optimal value
    const size_t acceptable_filter_x_size = 32;
    if (params.filterSize.x > acceptable_filter_x_size) {
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

KernelsData ConvolutionKernel_bfyx_GEMMLike::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector
