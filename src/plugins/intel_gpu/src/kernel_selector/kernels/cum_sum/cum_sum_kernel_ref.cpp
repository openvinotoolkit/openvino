// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
JitConstants CumSumKernelRef::GetJitConstants(const cum_sum_params& params, DispatchData dispatchData) const {
    auto jits = CumSumKernelBase::GetJitConstants(params, dispatchData);

    jits.AddConstant(MakeJitConstant("AXIS_LAYOUT_INDEX", GetCumSumAxisIndex(params)));

    return jits;
}

KernelsData CumSumKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority CumSumKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
