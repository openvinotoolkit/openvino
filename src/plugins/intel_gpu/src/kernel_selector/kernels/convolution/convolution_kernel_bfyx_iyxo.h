// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <string>

namespace kernel_selector {

class ConvolutionKernel_bfyx_iyxo : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_bfyx_iyxo() : Parent("convolution_gpu_bfyx_iyxo") {}
    virtual ~ConvolutionKernel_bfyx_iyxo() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override {
        return WeightsLayout::iyxo;
    }

    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p) const override;
    bool NeedPaddedInput() const override { return true; }
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
