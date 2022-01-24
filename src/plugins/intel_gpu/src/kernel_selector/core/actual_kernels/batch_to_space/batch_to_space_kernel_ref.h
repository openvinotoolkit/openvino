// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "batch_to_space_kernel_base.h"

namespace kernel_selector {
class BatchToSpaceKernelRef : public BatchToSpaceKernelBase {
public:
    using Parent = BatchToSpaceKernelBase;
    BatchToSpaceKernelRef() : BatchToSpaceKernelBase("batch_to_space_ref") {}
    virtual ~BatchToSpaceKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const batch_to_space_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
