// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class reorder_biplanar_nv12 : public ReorderKernelBase {
public:
    reorder_biplanar_nv12() : ReorderKernelBase("reorder_biplanar_nv12") {}
    virtual ~reorder_biplanar_nv12() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    JitConstants GetJitConstants(const reorder_params& params) const override;
};
}  // namespace kernel_selector
