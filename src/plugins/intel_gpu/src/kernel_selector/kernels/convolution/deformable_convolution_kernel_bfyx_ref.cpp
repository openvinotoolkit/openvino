// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deformable_convolution_kernel_bfyx_ref.h"

namespace kernel_selector {

ParamsKey DeformableConvolutionKernel_bfyx_Ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.DisableTuning();
    k.EnableGroupedConvolution();
    k.EnableDeformableMode();
    k.EnableDeformableMask();
    k.EnableBilinearInterpolationPad();
    return k;
}

KernelsData DeformableConvolutionKernel_bfyx_Ref::GetKernelsData(const Params& params,
                                                                 const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsPriority DeformableConvolutionKernel_bfyx_Ref::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool DeformableConvolutionKernel_bfyx_Ref::Validate(const Params& params, const optional_params& options) const {
    if (!ConvolutionKernelBase::Validate(params, options))
        return false;

    const auto& conv_params = static_cast<const convolution_params&>(params);

    if (!conv_params.deformable_mode)
        return false;

    return true;
}

}  // namespace kernel_selector
