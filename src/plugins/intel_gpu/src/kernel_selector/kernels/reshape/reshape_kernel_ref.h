// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reshape_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reshape_params : public base_params {
    reshape_params() : base_params(KernelType::RESHAPE) {}
};

class ReshapeKernelRef : public KernelBaseOpenCL {
public:
    ReshapeKernelRef() : KernelBaseOpenCL("reshape_ref") {}
    virtual ~ReshapeKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& pp) const override;
};
}  // namespace kernel_selector
