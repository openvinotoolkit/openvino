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

#include "weight_bias_kernel_base.h"

namespace kernel_selector {
JitConstants WeightBiasKernelBase::GetJitConstants(const weight_bias_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstants({
        MakeJitConstant("FILTER", params.weights),
        MakeJitConstant("BIAS_TERM", !params.bias.empty()),
    });

    if (params.bias.empty() == false) {
        const bool sameDims = params.bias[0].SameDims(params.output);
        jit.AddConstants({
            MakeJitConstant("BIAS", params.bias[0]),
            MakeJitConstant("BIAS_PER_OUTPUT", sameDims),
            MakeJitConstant("BIAS_PER_OFM", !sameDims),
        });
    }

    return jit;
}

}  // namespace kernel_selector