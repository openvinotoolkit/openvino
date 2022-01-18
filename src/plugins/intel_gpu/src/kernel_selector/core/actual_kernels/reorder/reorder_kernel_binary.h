// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernelBinary : public ReorderKernelBase {
public:
    ReorderKernelBinary() : ReorderKernelBase("reorder_data_binary") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const reorder_params& params) const override;
    DispatchData SetDefault(const reorder_params& arg) const override;

protected:
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
