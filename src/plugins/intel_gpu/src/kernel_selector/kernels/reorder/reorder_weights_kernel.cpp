// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector_common.h"
#include "reorder_weights_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderWeightsKernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::BF16);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT32);
    k.EnableOutputWeightsType(WeightsType::INT8);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::INT32);
    k.EnableAllInputWeightsLayout();
    k.EnableAllOutputWeightsLayout();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableRotateReorder();
    return k;
}

KernelsData ReorderWeightsKernel::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

JitConstants ReorderWeightsKernel::GetJitConstants(const reorder_weights_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    if ( params.input.GetDType() == WeightsType::BF16 ) {
        jit.AddConstant(MakeJitConstant("BF16_INPUT", true));
    }

    const auto output_layout = params.output.GetLayout();
    const bool has_imad_isv4_padding =
        (output_layout == WeightsLayout::os_is_yx_osv16_isv4 ||
         output_layout == WeightsLayout::g_os_is_yx_osv16_isv4) &&
        params.output.IFM().v % 4 != 0;
    jit.AddConstant(MakeJitConstant("IMAD_ISV4_PADDING", has_imad_isv4_padding));

    return jit;
}

ReorderWeightsKernel::DispatchData ReorderWeightsKernel::SetDefault(const reorder_weights_params& params) const {
    auto dispatch_data = ReorderKernelBase::SetDefault(params);
    const auto output_layout = params.output.GetLayout();

    if ((output_layout == WeightsLayout::os_is_yx_osv16_isv4 ||
         output_layout == WeightsLayout::g_os_is_yx_osv16_isv4) &&
        params.output.IFM().v % 4 != 0) {
        dispatch_data.gws[1] = Align(params.output.IFM().v, size_t{4});
        dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);
    }

    return dispatch_data;
}

KernelsPriority ReorderWeightsKernel::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
