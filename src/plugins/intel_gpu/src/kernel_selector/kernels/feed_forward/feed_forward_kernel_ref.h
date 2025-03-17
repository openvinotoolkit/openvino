// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// feed_forward_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct feed_forward_params : public base_params {
    feed_forward_params() : base_params(KernelType::FEED_FORWARD) {}
};


class FeedForwardKernelRef : public KernelBaseOpenCL {
public:
    FeedForwardKernelRef() : KernelBaseOpenCL("feed_forward_gpu_ref") {}
    virtual ~FeedForwardKernelRef() {}

    virtual JitConstants GetJitConstants(const feed_forward_params& params) const;
    virtual CommonDispatchData SetDefault(const feed_forward_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    // Datatype GetAccumulatorType(const feed_forward_params& params) const;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params&) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector