// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct eye_params : public base_params {
    eye_params() : base_params(KernelType::EYE) {}

    std::int32_t diagonal_index;
};

struct eye_optional_params : optional_params {
    eye_optional_params() : optional_params(KernelType::EYE) {}
};

class EyeKernelRef : public KernelBaseOpenCL {
public:
    EyeKernelRef() : KernelBaseOpenCL{"eye_ref"} {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p, const optional_params& o) const override;

private:
    JitConstants GetJitConstants(const eye_params& params) const;
    CommonDispatchData SetDefault(const eye_params& params, const optional_params&) const;
};

}  // namespace kernel_selector
