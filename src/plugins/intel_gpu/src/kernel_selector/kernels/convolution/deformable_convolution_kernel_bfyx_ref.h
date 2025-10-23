// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class DeformableConvolutionKernel_bfyx_Ref : public ConvolutionKernelBase {
public:
    DeformableConvolutionKernel_bfyx_Ref() : ConvolutionKernelBase("deformable_convolution_gpu_bfyx_ref") {}
    virtual ~DeformableConvolutionKernel_bfyx_Ref() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;

protected:
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& params) const override;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &params) const override {
        return (params.groups > 1) ? WeightsLayout::goiyx : WeightsLayout::oiyx;
    }
};
}  // namespace kernel_selector
