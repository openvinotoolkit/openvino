// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "deconvolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class DeconvolutionKernel_imad_ref : public DeconvolutionKernelBase {
public:
    using Parent = DeconvolutionKernelBase;
    DeconvolutionKernel_imad_ref() : DeconvolutionKernelBase("deconvolution_gpu_imad_ref") {}
    virtual ~DeconvolutionKernel_imad_ref() = default;

    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const deconvolution_params &params) const override;
    CommonDispatchData SetDefault(const deconvolution_params& params) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::QUANTIZE
        };
    }

    size_t GetTileIFM(const deconvolution_params& params) const;
};

}  // namespace kernel_selector
