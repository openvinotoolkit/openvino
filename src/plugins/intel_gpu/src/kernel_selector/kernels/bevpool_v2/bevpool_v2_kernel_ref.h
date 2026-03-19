// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "bevpool_v2_kernel_base.h"

namespace kernel_selector {

class BevPoolV2KernelRef : public BevPoolV2KernelBase {
public:
    BevPoolV2KernelRef() : BevPoolV2KernelBase("bevpool_v2_ref") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};

}  // namespace kernel_selector
