// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx : public ReorderKernelBase {
public:
    ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx() : ReorderKernelBase("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx") {}

    bool Validate(const Params& p) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
protected:
    JitConstants GetJitConstants(const reorder_params& params) const override;
    CommonDispatchData SetDefault(const reorder_params& params) const override;
};
}  // namespace kernel_selector
