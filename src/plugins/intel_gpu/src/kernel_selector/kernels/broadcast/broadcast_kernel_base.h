// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// broadcast_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct broadcast_params : public base_params {
    broadcast_params() : base_params(KernelType::BROADCAST) {}
    std::vector<uint16_t> input_order;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// broadcast_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct broadcast_optional_params : optional_params {
    broadcast_optional_params() : optional_params(KernelType::BROADCAST) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BroadcastKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BroadcastKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const broadcast_params& params) const;
    DispatchData SetDefault(const broadcast_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    // Tune params to the specific platform.
    const size_t vec_size = 2;
    const size_t y_blocks = 4;
};
}  // namespace kernel_selector
