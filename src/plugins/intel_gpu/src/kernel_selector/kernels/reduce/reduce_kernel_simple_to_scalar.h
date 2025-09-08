// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reduce_kernel_base.h"
#include <vector>

namespace kernel_selector {
class ReduceKernelSimpleToScalar : public ReduceKernelBase {
public:
    ReduceKernelSimpleToScalar() : ReduceKernelBase("reduce_simple_to_scalar") {}
    virtual ~ReduceKernelSimpleToScalar() {}
    CommonDispatchData SetDefault(const reduce_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const reduce_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ELTWISE,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
