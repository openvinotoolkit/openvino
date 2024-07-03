// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_params : public base_params {
    gather_params() : base_params(KernelType::GATHER), axis(GatherAxis::BATCH), batch_dim(0), support_neg_ind(false) {}

    GatherAxis axis;
    int64_t batch_dim;
    bool support_neg_ind;

    bool compressed = false;
    bool has_decompression_zp = false;
    bool scalar_zp = false;
    float zp_value = 0.0f;
    DataTensor decompression_scale;
    DataTensor decompression_zero_point;
};

class GatherKernelRef : public KernelBaseOpenCL {
public:
    GatherKernelRef() : KernelBaseOpenCL("gather_ref") {}
    virtual ~GatherKernelRef() {}
    virtual JitConstants GetJitConstants(const gather_params& params) const;
    virtual CommonDispatchData SetDefault(const gather_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ELTWISE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::REORDER };
    }

protected:
    bool Validate(const Params& p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
