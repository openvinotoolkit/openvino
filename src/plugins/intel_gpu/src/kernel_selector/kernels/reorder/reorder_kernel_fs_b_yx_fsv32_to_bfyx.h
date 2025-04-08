// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernel_fs_b_yx_fsv32_to_bfyx : public ReorderKernelBase {
public:
    ReorderKernel_fs_b_yx_fsv32_to_bfyx() : ReorderKernelBase("reorder_fs_b_yx_fsv32_to_bfyx") {}
    virtual ~ReorderKernel_fs_b_yx_fsv32_to_bfyx() {}

    DispatchData SetDefault(const reorder_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    JitConstants GetJitConstants(const reorder_params& params) const override;
};
}  // namespace kernel_selector
