// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "broadcast_kernel_base.h"

namespace kernel_selector {
class BroadcastKernelRef : public BroadcastKernelBase {
public:
    BroadcastKernelRef() : BroadcastKernelBase("broadcast_gpu_ref") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::REORDER };
    }
};
}  // namespace kernel_selector
