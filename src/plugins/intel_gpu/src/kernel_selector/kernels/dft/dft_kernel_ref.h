// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct dft_params : public base_params {
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    enum class Direction {
        forward,
        inverse,
    } direction = Direction::forward;
    enum class Mode {
        complex,
        real,
    } mode = Mode::complex;
    dft_params() : base_params{KernelType::DFT} {}
};

class DFTKernelRef : public KernelBaseOpenCL {
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const dft_params& params) const;

public:
    DFTKernelRef() : KernelBaseOpenCL{"dft_ref"} {}
};

}  // namespace kernel_selector
