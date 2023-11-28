// Copyright (C) 2018-2023 Intel Corporation
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

struct range_optional_params: optional_params {
    range_optional_params() :
        optional_params { KernelType::RANGE } {
    }
};

class RangeKernelRef: public KernelBaseOpenCL {
    KernelsData GetKernelsData(const Params &params, const optional_params &options) const override;
    KernelsPriority GetKernelsPriority(const Params &params, const optional_params &options) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params &p, const optional_params &o) const override;
    void SetUpdateDispatchDataFunc(KernelData& kd) const override;
public:
    RangeKernelRef() :
        KernelBaseOpenCL { "range_ref" } {
    }
};

}  // namespace kernel_selector
