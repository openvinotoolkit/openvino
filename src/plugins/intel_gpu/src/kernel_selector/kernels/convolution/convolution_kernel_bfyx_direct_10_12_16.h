// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_bfyx_Direct_10_10_12 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_bfyx_Direct_10_10_12() : ConvolutionKernelBase("convolution_gpu_bfyx_direct_10_12_16") {}
    virtual ~ConvolutionKernel_bfyx_Direct_10_10_12() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &p) const override {
        return (p.groups > 1) ? WeightsLayout::gi_yxs_os_yxsv2_osv16 : WeightsLayout::i_yxs_os_yxsv2_osv16;
    }

    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p) const override;
    bool NeedPaddedInput() const override { return true; }
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
