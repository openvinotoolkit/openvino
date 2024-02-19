// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include <vector>

namespace kernel_selector {

struct slice_params: public base_params {
    slice_params() : base_params(KernelType::SLICE) {}

    std::vector<std::int32_t> compile_time_start;
    std::vector<std::int32_t> compile_time_step;
    std::vector<std::int32_t> compile_time_axes;
    ov::element::Type_t start_data_type;
    ov::element::Type_t step_data_type;
    ov::element::Type_t axes_data_type;
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
    CommonDispatchData SetDefault(const slice_params &params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

} // namespace kernel_selector
