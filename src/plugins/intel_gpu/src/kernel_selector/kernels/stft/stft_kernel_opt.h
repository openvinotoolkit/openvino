// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "stft_kernel_base.h"

namespace kernel_selector {
class STFTKernelOpt : public STFTKernelBase {
public:
    STFTKernelOpt() : STFTKernelBase("stft_opt") {}

    JitConstants GetJitConstants(const STFT_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    CommonDispatchData CalcLaunchConfig(const STFT_params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector
