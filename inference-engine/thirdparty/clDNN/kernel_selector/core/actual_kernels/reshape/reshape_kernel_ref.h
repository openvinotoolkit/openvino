// Copyright (C) 2018-2021 Intel Corporation
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reshape_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reshape_optional_params : optional_params {
    reshape_optional_params() : optional_params(KernelType::RESHAPE) {}
};

class ReshapeKernelRef : public KernelBaseOpenCL {
public:
    ReshapeKernelRef() : KernelBaseOpenCL("reshape_ref") {}
    virtual ~ReshapeKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& op) const override;
};
}  // namespace kernel_selector
