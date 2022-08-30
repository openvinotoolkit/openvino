// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_gemv_gpu_subgroup1x64_bfyx_hh_simd16.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey LSTMGemvKernel_subgroup1x64_bfyx_hh_SIMD16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLSTMGEMMBias();
    k.EnableLSTMGEMMHidden();
    k.EnableSubGroup();
    return k;
}

KernelsData LSTMGemvKernel_subgroup1x64_bfyx_hh_SIMD16::GetKernelsData(const Params& params,
                                                                       const optional_params& options) const {
    KernelsData kernelsData = GetCommonKernelsData(params, options);
    auto& kernel = kernelsData[0].kernels[0];

    // This kernel is good if
    // 1) Batch size is 1
    // 2) The input size y-x size is 64x1
    const lstm_gemm_params& orgParams = static_cast<const lstm_gemm_params&>(params);
    const auto& input = orgParams.inputs[0];
    const auto& out = orgParams.outputs[0];

    if ((input.Batch().v == 1) && (input.X().v >= 64) && (input.Y().v == 1))
        kernel.params.workGroups.global = {16, out.X().v, out.Batch().v};

    return kernelsData;
}

KernelsPriority LSTMGemvKernel_subgroup1x64_bfyx_hh_SIMD16::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    const lstm_gemm_params& orgParams = static_cast<const lstm_gemm_params&>(params);
    const auto& input = orgParams.inputs[0];

    if ((input.Batch().v == 1) && (input.X().v >= 64) && (input.Y().v == 1))
        return FORCE_PRIORITY_1;
    else
        return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
