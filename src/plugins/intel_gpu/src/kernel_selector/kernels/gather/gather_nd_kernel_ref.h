// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_nd_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_nd_params : public base_params {
    gather_nd_params() : base_params(KernelType::GATHER_ND), indices_rank(0), batch_dims(0), batch_merged_output(true) {}

    uint8_t indices_rank;

    uint8_t batch_dims;

    bool batch_merged_output;
};

class GatherNDKernelRef : public KernelBaseOpenCL {
public:
    GatherNDKernelRef() : KernelBaseOpenCL("gather_nd_ref") {}
    virtual ~GatherNDKernelRef() {}
    virtual JitConstants GetJitConstants(const gather_nd_params& params) const;
    virtual CommonDispatchData SetDefault(const gather_nd_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
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
