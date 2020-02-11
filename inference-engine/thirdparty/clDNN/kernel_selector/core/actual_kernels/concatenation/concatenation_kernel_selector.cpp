// Copyright (c) 2016 Intel Corporation
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


#include "concatenation_kernel_selector.h"
#include "concatenation_kernel_ref.h"
#include "concatenation_kernel_simple_ref.h"
#include "concatenation_kernel_depth_bfyx_no_pitch.h"
#include "concatenation_kernel_blocked.h"
#include "concatenation_kernel_fs_b_yx_fsv32.h"

namespace kernel_selector {
concatenation_kernel_selector::concatenation_kernel_selector() {
    Attach<ConcatenationKernelRef>();
    Attach<ConcatenationKernel_simple_Ref>();
    Attach<ConcatenationKernel_depth_bfyx_no_pitch>();
    Attach<ConcatenationKernelBlocked>();
    Attach<ConcatenationKernel_fs_b_yx_fsv32>();
}

KernelsData concatenation_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::CONCATENATION);
}
}  // namespace kernel_selector
