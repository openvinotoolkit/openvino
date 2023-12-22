// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "kernel_base_opencl.h"

namespace kernel_selector {

struct roll_params : base_params {
    roll_params() : base_params(KernelType::ROLL) {}
    DimTensor<> shift;
};

struct roll_optional_params : optional_params {
    roll_optional_params() : optional_params(KernelType::ROLL) {}
};

class RollKernelRef : public KernelBaseOpenCL {
public:
    RollKernelRef() : KernelBaseOpenCL{"roll_ref"} {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const roll_params& kernel_params) const;
};

}  // namespace kernel_selector
