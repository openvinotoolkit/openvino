// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct scaled_shifted_clamp_experimental_params : public base_params {
    scaled_shifted_clamp_experimental_params()
        : base_params(KernelType::SCALED_SHIFTED_CLAMP_EXPERIMENTAL) {}
    float scale{1.0f};
    float bias{0.0f};
    float lo{0.0f};
    float hi{0.0f};
};

class ScaledShiftedClampExperimentalKernelRef : public KernelBaseOpenCL {
public:
    ScaledShiftedClampExperimentalKernelRef() : KernelBaseOpenCL("scaled_shifted_clamp_experimental_ref") {}

    KernelsData    GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey      GetSupportedKey() const override;
    bool           Validate(const Params& p) const override;
    void           GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
