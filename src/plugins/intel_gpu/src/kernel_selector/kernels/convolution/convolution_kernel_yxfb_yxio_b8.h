// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_yxfb_yxio_b8 : public ConvolutionKernelBase {
public:
    ConvolutionKernel_yxfb_yxio_b8() : ConvolutionKernelBase("convolution_gpu_yxfb_yxio_b8_fp32") {}
    virtual ~ConvolutionKernel_yxfb_yxio_b8() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::yxio;
    }
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
