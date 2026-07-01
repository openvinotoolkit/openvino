// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct histc_params : base_params {
    histc_params() : base_params(KernelType::HISTC) {}
    int64_t bins = 100;
    double min_val = 0.0;
    double max_val = 0.0;
};

class HistcKernelRef : public KernelBaseOpenCL {
public:
    HistcKernelRef() : KernelBaseOpenCL{"histc_ref"} {}

private:
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const histc_params& kernel_params) const;
};

}  // namespace kernel_selector
