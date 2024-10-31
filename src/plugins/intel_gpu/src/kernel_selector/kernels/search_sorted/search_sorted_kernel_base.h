// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// search_sorted
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct search_sorted_params : public base_params {
    search_sorted_params() : base_params(KernelType::ONE_HOT),
    one_hot_axis(0), one_hot_limit(0), on_value(1.0), off_value(1.0) {}
    uint16_t one_hot_axis;
    int32_t one_hot_limit;
    float on_value;
    float off_value;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SearchSortedKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SearchSortedKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const search_sorted_params& params) const;
    static DispatchData SetDefault(const search_sorted_params& params);
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
