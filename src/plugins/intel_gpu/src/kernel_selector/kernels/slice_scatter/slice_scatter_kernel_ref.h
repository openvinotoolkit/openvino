// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include <vector>

namespace kernel_selector {

struct slice_scatter_params : public base_params {
    slice_scatter_params() : base_params(KernelType::SLICE_SCATTER) {}

    std::vector<std::int64_t> compile_time_start;
    std::vector<std::int64_t> compile_time_step;
    std::vector<std::int64_t> compile_time_axes;
    kernel_selector::Datatype start_data_type = kernel_selector::Datatype::UNSUPPORTED;
    kernel_selector::Datatype step_data_type = kernel_selector::Datatype::UNSUPPORTED;
    kernel_selector::Datatype axes_data_type = kernel_selector::Datatype::UNSUPPORTED;
};

class SliceScatterKernelRef : public KernelBaseOpenCL {
public:
    SliceScatterKernelRef() : KernelBaseOpenCL{"slice_scatter_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p) const override;

private:
    JitConstants GetJitConstants(const slice_scatter_params& params) const;
    CommonDispatchData SetDefault(const slice_scatter_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
