// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "swiglu_kernel_base.h"

namespace kernel_selector {
class SwiGLUKernelOpt : public SwiGLUKernelBase {
public:
    SwiGLUKernelOpt() : SwiGLUKernelBase("swiglu_gpu_opt") {}
    virtual ~SwiGLUKernelOpt() {}

protected:
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
