// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "deconvolution_kernel_base.h"

namespace kernel_selector {

class DeconvolutionKernelRef : public DeconvolutionKernelBase {
public:
    DeconvolutionKernelRef() : DeconvolutionKernelBase("deconvolution_gpu_ref") {}
    virtual ~DeconvolutionKernelRef() {}

    ParamsKey GetSupportedKey() const override;

protected:
    CommonDispatchData SetDefault(const deconvolution_params& params) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
