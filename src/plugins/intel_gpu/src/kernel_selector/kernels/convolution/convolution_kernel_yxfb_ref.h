// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_yxfb_Ref : public ConvolutionKernelBase {
public:
    ConvolutionKernel_yxfb_Ref() : ConvolutionKernelBase("convolution_gpu_yxfb_ref") {}
    virtual ~ConvolutionKernel_yxfb_Ref() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &params) const override {
        return (params.groups > 1) ? WeightsLayout::gyxio: WeightsLayout::yxio;
    }
};
}  // namespace kernel_selector
