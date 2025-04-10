// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "concatenation_kernel_base.h"

namespace kernel_selector {

class ConcatenationKernel_depth_bfyx_no_pitch : public ConcatenationKernelBase {
public:
    ConcatenationKernel_depth_bfyx_no_pitch() : ConcatenationKernelBase("concatenation_gpu_depth_bfyx_no_pitch") {}
    virtual ~ConcatenationKernel_depth_bfyx_no_pitch() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    DispatchData SetDefault(const concatenation_params& params) const override;
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector
