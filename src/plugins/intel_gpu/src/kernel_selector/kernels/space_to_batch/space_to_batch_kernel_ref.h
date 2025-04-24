// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "space_to_batch_kernel_base.h"

namespace kernel_selector {
class SpaceToBatchKernelRef : public SpaceToBatchKernelBase {
public:
    using Parent = SpaceToBatchKernelBase;
    SpaceToBatchKernelRef() : SpaceToBatchKernelBase("space_to_batch_ref") {}
    virtual ~SpaceToBatchKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const space_to_batch_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
