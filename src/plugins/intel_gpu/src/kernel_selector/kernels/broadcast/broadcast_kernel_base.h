// Copyright (C) 2018-2025 Intel Corporation
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
// BroadcastKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BroadcastKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const broadcast_params& params) const;
    static DispatchData SetDefault(const broadcast_params& params);
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
