// Copyright (C) 2018-2025 Intel Corporation
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
    search_sorted_params() : base_params(KernelType::SEARCH_SORTED), right_mode(false) {}
    bool right_mode;
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
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
