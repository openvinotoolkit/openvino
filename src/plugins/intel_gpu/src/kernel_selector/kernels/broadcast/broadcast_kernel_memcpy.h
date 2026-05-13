// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "broadcast_kernel_base.h"

namespace kernel_selector {
class BroadcastKernelMemcpy : public BroadcastKernelBase {
public:
    BroadcastKernelMemcpy() : BroadcastKernelBase("broadcast_gpu_memcpy") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& params) const override;
};
}  // namespace kernel_selector
