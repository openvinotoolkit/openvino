// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "deconvolution_kernel_base.h"

namespace kernel_selector {

class DeconvolutionKernel_bfyx_opt : public DeconvolutionKernelBase {
public:
    using Parent = DeconvolutionKernelBase;
    DeconvolutionKernel_bfyx_opt() : DeconvolutionKernelBase("deconvolution_gpu_bfyx_opt") {}
    virtual ~DeconvolutionKernel_bfyx_opt() {}

    ParamsKey GetSupportedKey() const override;

protected:
    CommonDispatchData SetDefault(const deconvolution_params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
