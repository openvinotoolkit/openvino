// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "concatenation_kernel_base.h"

namespace kernel_selector {

class ConcatenationKernelRef : public ConcatenationKernelBase {
public:
    ConcatenationKernelRef() : ConcatenationKernelBase("concatenation_gpu_ref") {}
    virtual ~ConcatenationKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const concatenation_params& params) const override;
};
}  // namespace kernel_selector