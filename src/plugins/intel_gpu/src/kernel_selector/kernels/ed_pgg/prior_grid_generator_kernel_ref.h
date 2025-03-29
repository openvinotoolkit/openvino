// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * ExperimentalDetectronPriorGridGenerator reference kernel parameters
 */
struct experimental_detectron_prior_grid_generator_params : public base_params {
    experimental_detectron_prior_grid_generator_params()
        : base_params{KernelType::EXPERIMENTAL_DETECTRON_PRIOR_GRID_GENERATOR} {}

    bool flatten = false;
    size_t layer_height = 0;
    size_t layer_width = 0;
    float step_x = 0;
    float step_y = 0;
};

/**
 * Reference kernel for ExperimentalDetectronPriorGridGenerator
 */
class ExperimentalDetectronPriorGridGeneratorKernelRef : public KernelBaseOpenCL {
public:
    ExperimentalDetectronPriorGridGeneratorKernelRef()
        : KernelBaseOpenCL{"experimental_detectron_prior_grid_generator_ref"} {}

private:
    KernelsData GetKernelsData(const Params &params) const override;

    KernelsPriority GetKernelsPriority(const Params &params) const override;

    ParamsKey GetSupportedKey() const override;

    bool Validate(const Params &p) const override;

    JitConstants GetJitConstants(const experimental_detectron_prior_grid_generator_params &params) const;
};

}  // namespace kernel_selector
