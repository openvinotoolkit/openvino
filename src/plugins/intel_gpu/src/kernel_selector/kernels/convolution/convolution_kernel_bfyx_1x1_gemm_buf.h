// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_bfyx_1x1_gemm_buf : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;

    ConvolutionKernel_bfyx_1x1_gemm_buf() : ConvolutionKernelBase("convolution_gpu_bfyx_1x1_hgemm_buf_16x1") {}
    virtual ~ConvolutionKernel_bfyx_1x1_gemm_buf() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::image_2d_weights_c1_b_fyx;
    }
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
};
}  // namespace kernel_selector
