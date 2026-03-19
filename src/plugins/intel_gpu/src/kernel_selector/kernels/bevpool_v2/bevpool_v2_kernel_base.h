// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {

struct bevpool_v2_params : public base_params {
    bevpool_v2_params() : base_params(KernelType::BEVPOOL_V2) {}

    uint32_t input_channels = 0;
    uint32_t output_channels = 0;
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    uint32_t feature_width = 0;
    uint32_t feature_height = 0;

    float x_bound_min = 0.f;
    float x_bound_max = 0.f;
    float x_bound_step = 1.f;

    float y_bound_min = 0.f;
    float y_bound_max = 0.f;
    float y_bound_step = 1.f;

    float z_bound_min = 0.f;
    float z_bound_max = 0.f;
    float z_bound_step = 1.f;

    float d_bound_min = 0.f;
    float d_bound_max = 0.f;
    float d_bound_step = 1.f;
};

class BevPoolV2KernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const bevpool_v2_params& params) const;
    static DispatchData SetDefault(const bevpool_v2_params& params);
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
