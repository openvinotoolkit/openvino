// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsBF16 : public ReorderKernelBase {
public:
    ReorderWeightsBF16() : ReorderKernelBase("reorder_weights_bf16") {}
    virtual ~ReorderWeightsBF16() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector