// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernel_to_yxfb_batched : public ReorderKernelBase {
public:
    ReorderKernel_to_yxfb_batched() : ReorderKernelBase("reorder_data_to_yxfb_batched") {}
    ParamsKey GetSupportedKey() const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    JitConstants GetJitConstants(const reorder_params& params) const override;
    DispatchData SetDefault(const reorder_params& arg) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector