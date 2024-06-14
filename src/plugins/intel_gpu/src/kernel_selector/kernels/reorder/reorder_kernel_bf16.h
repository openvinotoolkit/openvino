// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernelBF16 : public ReorderKernelBase {
public:
    ReorderKernelBF16() : ReorderKernelBase("reorder_data_bf16") {}
    virtual ~ReorderKernelBF16() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    JitConstants GetJitConstants(const reorder_params& params) const override;
};
}  // namespace kernel_selector
