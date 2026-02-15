// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * GridSample reference kernel parameters.
 */
struct grid_sample_params : public base_params {
    grid_sample_params() : base_params(KernelType::GRID_SAMPLE) {}
    bool align_corners = false;
    enum class InterpolationMode {
        BILINEAR,
        BICUBIC,
        NEAREST,
    } interpolation_mode = InterpolationMode::BILINEAR;
    enum class PaddingMode {
        ZEROS,
        BORDER,
        REFLECTION,
    } padding_mode = PaddingMode::ZEROS;
};

/**
 * GridSampleKernelBase.
 */
class GridSampleKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    KernelsData GetKernelsData(const Params& params) const override;

protected:
    virtual CommonDispatchData CalcDispatch(const grid_sample_params& kernel_params) const = 0;
    virtual JitConstants GetJitConstants(const grid_sample_params& kernel_params) const;

    bool Validate(const Params& params) const override;
};

}  // namespace kernel_selector
