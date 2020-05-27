/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "cum_sum_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
JitConstants CumSumKernelRef::GetJitConstants(const cum_sum_params& params, DispatchData kd) const {
    auto jits = CumSumKernelBase::GetJitConstants(params, kd);

    jits.AddConstant(MakeJitConstant("AXIS_LAYOUT_INDEX", GetCumSumAxisIndex(params)));

    return jits;
}

KernelsData CumSumKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, DONT_USE_IF_HAVE_SOMETHING_ELSE);
}
}  // namespace kernel_selector
