// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_nonzero_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_nonzero_params : public base_params {
    gather_nonzero_params() : base_params(KernelType::GATHER_NONZERO) {}
    int32_t ov_input_rank = -1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_nonzero_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_nonzero_optional_params : optional_params {
    gather_nonzero_optional_params() : optional_params(KernelType::GATHER_NONZERO) {}
};

class GatherNonzeroKernelRef : public KernelBaseOpenCL {
public:
    GatherNonzeroKernelRef() : KernelBaseOpenCL("gather_nonzero_ref") {}
    virtual ~GatherNonzeroKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& op) const override;
};
}  // namespace kernel_selector
