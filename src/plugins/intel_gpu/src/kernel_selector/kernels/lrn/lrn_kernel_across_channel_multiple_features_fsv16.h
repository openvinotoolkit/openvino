// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lrn_kernel_across_channel_multiple_features.h"

namespace kernel_selector {
class LRNKernelAcrossChannelMultipleFeaturesFSV16 : public LRNKernelAcrossChannelMultipleFeatures {
public:
    using Parent = LRNKernelAcrossChannelMultipleFeatures;
    LRNKernelAcrossChannelMultipleFeaturesFSV16() : Parent("lrn_gpu_across_channel_multiple_features_fsv16") {}

    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;

private:
    DispatchData SetDefault(const lrn_params& params) const override;
    JitConstants GetJitConstants(const lrn_params& params, const DispatchData& dispatchData) const override;
};
}  // namespace kernel_selector
