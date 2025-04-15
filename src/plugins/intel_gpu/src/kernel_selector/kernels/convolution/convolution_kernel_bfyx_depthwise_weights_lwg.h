// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {
class ConvolutionKernel_bfyx_depthwise_weights_lwg : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_bfyx_depthwise_weights_lwg()
        : ConvolutionKernelBase("convolution_gpu_bfyx_depthwise_weights_lwg") {}
    virtual ~ConvolutionKernel_bfyx_depthwise_weights_lwg() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params&) const override;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::goiyx;
    }
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
