// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_elements_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_elements_params : public base_params {
    gather_elements_params() : base_params(KernelType::GATHER_ELEMENTS), axis(GatherAxis::BATCH) {}

    GatherAxis axis;
};

class GatherElementsKernelRef : public KernelBaseOpenCL {
public:
    GatherElementsKernelRef() : KernelBaseOpenCL("gather_elements_ref") {}
    virtual ~GatherElementsKernelRef() {}
    virtual JitConstants GetJitConstants(const gather_elements_params& params) const;
    virtual CommonDispatchData SetDefault(const gather_elements_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }

protected:
    bool Validate(const Params& p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
