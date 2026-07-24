// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "broadcast_kernel_base.h"

namespace kernel_selector {
class BroadcastKernelOpt : public BroadcastKernelBase {
public:
    BroadcastKernelOpt() : BroadcastKernelBase("broadcast_gpu_opt") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& params) const override;

protected:
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
