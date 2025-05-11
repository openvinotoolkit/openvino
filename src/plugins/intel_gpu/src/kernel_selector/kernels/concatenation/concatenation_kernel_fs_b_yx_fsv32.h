// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "concatenation_kernel_base.h"

namespace kernel_selector {

class ConcatenationKernel_fs_b_yx_fsv32 : public ConcatenationKernelBase {
public:
    ConcatenationKernel_fs_b_yx_fsv32() : ConcatenationKernelBase("concatenation_gpu_fs_b_yx_fsv32") {}
    virtual ~ConcatenationKernel_fs_b_yx_fsv32() {}

    KernelsData GetKernelsData(const Params& params) const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const concatenation_params& params) const override;
    JitConstants GetJitConstants(const concatenation_params& params) const override;
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector
