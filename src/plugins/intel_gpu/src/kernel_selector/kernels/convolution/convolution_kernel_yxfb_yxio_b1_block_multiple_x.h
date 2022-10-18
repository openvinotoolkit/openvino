// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_yxfb_yxio_b1_block_multiple_x : public ConvolutionKernelBase {
public:
    ConvolutionKernel_yxfb_yxio_b1_block_multiple_x()
        : ConvolutionKernelBase("convolution_gpu_yxfb_yxio_b1_block_multiple_x_fp32") {}
    virtual ~ConvolutionKernel_yxfb_yxio_b1_block_multiple_x() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::yxio;
    }
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
