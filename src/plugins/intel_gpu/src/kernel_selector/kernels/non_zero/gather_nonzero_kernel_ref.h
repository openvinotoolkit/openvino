// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "gather_nonzero_params.h"

namespace kernel_selector {
class GatherNonzeroKernelRef : public KernelBaseOpenCL {
public:
    GatherNonzeroKernelRef() : KernelBaseOpenCL("gather_nonzero_ref") {}
    virtual ~GatherNonzeroKernelRef() {}

    virtual JitConstants GetJitConstants(const gather_nonzero_params& params) const;
    virtual CommonDispatchData SetDefault(const gather_nonzero_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& pp) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
