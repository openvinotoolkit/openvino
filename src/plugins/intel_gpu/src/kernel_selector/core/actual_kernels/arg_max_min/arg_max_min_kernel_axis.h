// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arg_max_min_kernel_base.h"

namespace kernel_selector {
class ArgMaxMinKernelAxis : public ArgMaxMinKernelBase {
public:
    ArgMaxMinKernelAxis() : ArgMaxMinKernelBase("arg_max_min_axis") {}
    virtual ~ArgMaxMinKernelAxis() {}

    JitConstants GetJitConstants(const arg_max_min_params& params) const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
private:
    bool Validate(const Params&, const optional_params&) const override;
};
}  // namespace kernel_selector