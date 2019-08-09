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

    if ((input.Batch().v == 1) && (input.X().v >= 64) && (input.Y().v == 1)) {
        auto out = orgParams.output;

        kernel.workGroups.global = {16, out.X().v, out.Batch().v};
        kernelsData[0].estimatedTime = FORCE_PRIORITY_1;
    }

    return kernelsData;
}
}  // namespace kernel_selector