// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "concatenation_kernel_base.h"

namespace kernel_selector {

class ConcatenationKernelRef : public ConcatenationKernelBase {
public:
    ConcatenationKernelRef() : ConcatenationKernelBase("concatenation_gpu_ref") {}
    virtual ~ConcatenationKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const concatenation_params& params) const override;
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
