// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_update_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scatter_update_params : public base_params {
    scatter_update_params() : base_params(KernelType::SCATTER_UPDATE), axis(ScatterUpdateAxis::BATCH) {}

    ScatterUpdateAxis axis;
};

class ScatterUpdateKernelRef : public KernelBaseOpenCL {
public:
    ScatterUpdateKernelRef() : KernelBaseOpenCL("scatter_update_ref") {}
    virtual ~ScatterUpdateKernelRef() {}
    virtual JitConstants GetJitConstants(const scatter_update_params& params) const;
    virtual CommonDispatchData SetDefault(const scatter_update_params& params, bool is_second) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

protected:
    bool Validate(const Params& p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
