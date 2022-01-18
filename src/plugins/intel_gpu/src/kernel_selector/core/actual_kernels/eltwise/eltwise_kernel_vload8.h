// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise_kernel_base.h"

namespace kernel_selector {
class EltwiseKernel_vload8 : public EltwiseKernelBase {
public:
    EltwiseKernel_vload8() : EltwiseKernelBase("eltwise_simple_vload8") {}
    virtual ~EltwiseKernel_vload8() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
};
}  // namespace kernel_selector