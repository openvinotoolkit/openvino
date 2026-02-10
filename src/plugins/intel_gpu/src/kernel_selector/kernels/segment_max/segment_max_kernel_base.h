// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// segment_max_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct segment_max_params : public base_params {
    segment_max_params() : base_params(KernelType::SEGMENT_MAX), fill_mode(0) {}

    // fill_mode: 0 = ZERO, 1 = LOWEST
    int fill_mode;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SegmentMaxKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SegmentMaxKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const segment_max_params& params) const;
    static DispatchData SetDefault(const segment_max_params& params);
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
