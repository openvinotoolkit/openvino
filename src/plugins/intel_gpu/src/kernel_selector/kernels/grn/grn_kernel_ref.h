// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "grn_kernel_base.h"

namespace kernel_selector {
class GRNKernelRef : public GRNKernelBase {
public:
    using Parent = GRNKernelBase;
    GRNKernelRef() : GRNKernelBase("grn_ref") {}
    virtual ~GRNKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
