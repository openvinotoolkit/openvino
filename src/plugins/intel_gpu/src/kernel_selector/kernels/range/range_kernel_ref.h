// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct range_params: public base_params {
    range_params() :
        base_params { KernelType::RANGE } {
    }
};

class RangeKernelRef: public KernelBaseOpenCL {
    KernelsData GetKernelsData(const Params &params) const override;
    KernelsPriority GetKernelsPriority(const Params &params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params &p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
public:
    RangeKernelRef() :
        KernelBaseOpenCL { "range_ref" } {
    }
};

}  // namespace kernel_selector
