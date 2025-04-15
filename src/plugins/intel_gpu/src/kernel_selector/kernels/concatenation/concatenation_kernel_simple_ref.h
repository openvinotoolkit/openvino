// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "concatenation_kernel_base.h"

namespace kernel_selector {

class ConcatenationKernel_simple_Ref : public ConcatenationKernelBase {
public:
    ConcatenationKernel_simple_Ref() : ConcatenationKernelBase("concatenation_gpu_simple_ref") {}
    virtual ~ConcatenationKernel_simple_Ref() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    JitConstants GetJitConstants(const concatenation_params& params) const override;
    DispatchData SetDefault(const concatenation_params& params) const override;
    bool Validate(const Params& p) const override;

protected:
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::REORDER,
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
