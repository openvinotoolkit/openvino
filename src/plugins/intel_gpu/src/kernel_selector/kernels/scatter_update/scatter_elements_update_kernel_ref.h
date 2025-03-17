// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_elements_update_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scatter_elements_update_params : public base_params {
    scatter_elements_update_params() : base_params(KernelType::SCATTER_ELEMENTS_UPDATE) {}

    ScatterUpdateAxis axis{ScatterUpdateAxis::BATCH};
    ScatterUpdateReduction mode{ScatterUpdateReduction::NONE};
    bool use_init_val{true};
};

class ScatterElementsUpdateKernelRef : public KernelBaseOpenCL {
public:
    ScatterElementsUpdateKernelRef() : KernelBaseOpenCL("scatter_elements_update_ref") {}
    virtual ~ScatterElementsUpdateKernelRef() {}
    virtual JitConstants GetJitConstants(const scatter_elements_update_params& params) const;
    virtual CommonDispatchData SetDefault(const scatter_elements_update_params& params, bool is_second) const;
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }

protected:
    bool Validate(const Params& p) const override;
    bool SkipKernelExecution(const scatter_elements_update_params& params, size_t kernel_id) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
