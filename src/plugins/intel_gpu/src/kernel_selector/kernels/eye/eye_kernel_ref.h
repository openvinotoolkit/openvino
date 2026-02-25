// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct eye_params : public base_params {
    eye_params() : base_params(KernelType::EYE) {}

    std::int32_t diagonal_index = 0;
};

class EyeKernelRef : public KernelBaseOpenCL {
public:
    EyeKernelRef() : KernelBaseOpenCL{"eye_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p) const override;

private:
    JitConstants GetJitConstants(const eye_params& params) const;
    CommonDispatchData SetDefault(const eye_params& params) const;
};

}  // namespace kernel_selector
