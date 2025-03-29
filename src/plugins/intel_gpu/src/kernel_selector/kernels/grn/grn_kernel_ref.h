// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "grn_kernel_base.h"
#include <string>
#include <vector>

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
