// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// strided_slice_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct strided_slice_params : public base_params {
    strided_slice_params() : base_params(KernelType::STRIDED_SLICE) {}

    std::vector<std::vector<int32_t>> striding_params;
    std::vector<uint8_t> begin_mask;
    std::vector<uint8_t> end_mask;
    std::vector<uint8_t> ellipsis_mask;
    std::vector<uint8_t> new_axis_mask;
    std::vector<uint8_t> shrink_axis_mask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// strided_slice_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct strided_slice_optional_params : optional_params {
    strided_slice_optional_params() : optional_params(KernelType::STRIDED_SLICE) {}
};

class StridedSliceKernelRef : public KernelBaseOpenCL {
public:
    StridedSliceKernelRef() : KernelBaseOpenCL("strided_slice_ref") {}
    virtual ~StridedSliceKernelRef() {}
    virtual JitConstants GetJitConstants(const strided_slice_params& params) const;
    virtual CommonDispatchData SetDefault(const strided_slice_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
