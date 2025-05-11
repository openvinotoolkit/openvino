// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * Random Uniform kernel params. All the needed inputs are in base input, output shape is static and
 * presented in output.
 */
struct random_uniform_params: public base_params {
    random_uniform_params() :
        base_params{KernelType::RANDOM_UNIFORM} {
    }

    // operation attributes
    uint64_t global_seed = 0;
    uint64_t op_seed = 0;
};

/**
 * Reference GPU kernel for the RandomUniform-8 operation.
 */
class RandomUniformKernelRef: public KernelBaseOpenCL {
public:
    RandomUniformKernelRef() :
        KernelBaseOpenCL{"random_uniform_ref"} {
    }
private:
    KernelsData GetKernelsData(const Params &params) const override;

    KernelsPriority GetKernelsPriority(const Params &params) const override;

    ParamsKey GetSupportedKey() const override;

    bool Validate(const Params &params) const override;

    JitConstants GetJitConstants(const random_uniform_params &params) const;
};

} /* namespace kernel_selector */
