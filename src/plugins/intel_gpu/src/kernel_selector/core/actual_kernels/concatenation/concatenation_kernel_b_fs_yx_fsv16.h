// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "concatenation_kernel_base.h"

namespace kernel_selector {

class ConcatenationKernel_b_fs_yx_fsv16 : public ConcatenationKernelBase {
public:
    ConcatenationKernel_b_fs_yx_fsv16() : ConcatenationKernelBase("concatenation_gpu_blocked") {}
    virtual ~ConcatenationKernel_b_fs_yx_fsv16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const concatenation_params& params) const override;
    JitConstants GetJitConstants(const concatenation_params& params) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    size_t GetAlignment(const concatenation_params& params) const override;
};
}  // namespace kernel_selector
