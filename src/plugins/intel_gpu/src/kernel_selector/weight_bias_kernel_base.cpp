// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weight_bias_kernel_base.h"

namespace kernel_selector {
JitConstants WeightBiasKernelBase::GetJitConstants(const weight_bias_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstants({
        MakeJitConstant("FILTER", params.weights),
        MakeJitConstant("BIAS_TERM", !params.bias.empty()),
    });

    if (params.bias.empty() == false) {
        const bool sameDims = params.bias[0].SameDims(params.outputs[0]);
        jit.AddConstants({
            MakeJitConstant("BIAS", params.bias[0]),
            MakeJitConstant("BIAS_PER_OUTPUT", sameDims),
            MakeJitConstant("BIAS_PER_OFM", !sameDims),
        });
    }

    return jit;
}

}  // namespace kernel_selector
