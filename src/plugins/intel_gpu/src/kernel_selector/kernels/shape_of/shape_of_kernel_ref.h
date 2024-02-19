// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct shape_of_params: public base_params {
    shape_of_params() :
        base_params { KernelType::SHAPE_OF } {
    }

    size_t input_rank = 0;
    std::vector<int32_t> input_dims = {};
};

class ShapeOfKernelRef: public KernelBaseOpenCL {
    KernelsData GetKernelsData(const Params &params) const override;
    KernelsPriority GetKernelsPriority(const Params &params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params &p) const override;
    virtual JitConstants GetJitConstants(const shape_of_params& params) const;
    bool SkipKernelExecution(const shape_of_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
public:
    ShapeOfKernelRef() :
        KernelBaseOpenCL { "shape_of_ref" } {
    }
};

}  // namespace kernel_selector
