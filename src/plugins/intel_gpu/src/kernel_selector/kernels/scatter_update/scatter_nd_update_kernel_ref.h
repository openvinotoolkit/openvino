// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_nd_update_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scatter_nd_update_params : public base_params {
    scatter_nd_update_params() : base_params(KernelType::SCATTER_ND_UPDATE), indices_rank(0) {}

    size_t indices_rank;
};

class ScatterNDUpdateKernelRef : public KernelBaseOpenCL {
public:
    struct DispatchData : public CommonDispatchData {
        size_t indicesLastDim;
    };

    ScatterNDUpdateKernelRef() : KernelBaseOpenCL("scatter_nd_update_ref") {}
    virtual ~ScatterNDUpdateKernelRef() {}
    virtual JitConstants GetJitConstants(const scatter_nd_update_params& params) const;
    virtual DispatchData SetDefault(const scatter_nd_update_params& params, bool is_second) const;
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
