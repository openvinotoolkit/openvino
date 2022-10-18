// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include <vector>

namespace kernel_selector {

struct slice_params: public base_params {
    slice_params() : base_params(KernelType::SLICE) {}

    std::vector<std::int32_t> start;
    std::vector<std::int32_t> end;
    std::vector<std::int32_t> step;
};

struct slice_optional_params : optional_params {
    slice_optional_params() : optional_params(KernelType::SLICE) {}
};

class SliceKernelRef: public KernelBaseOpenCL {
public:
    SliceKernelRef() :
            KernelBaseOpenCL { "slice_ref" } {
    }
    KernelsData GetKernelsData(const Params &params,
            const optional_params &options) const override;
    KernelsPriority GetKernelsPriority(const Params &params,
            const optional_params &options) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params &p, const optional_params &o) const override;

private:
    JitConstants GetJitConstants(const slice_params &params) const;
    CommonDispatchData SetDefault(const slice_params &params,
            const optional_params&) const;
};

} // namespace kernel_selector
