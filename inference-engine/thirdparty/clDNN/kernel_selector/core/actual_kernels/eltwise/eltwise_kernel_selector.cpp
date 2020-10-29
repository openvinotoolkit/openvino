// Copyright (c) 2019 Intel Corporation
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


#include "eltwise_kernel_selector.h"
#include "eltwise_kernel_ref.h"
#include "eltwise_kernel_vload8.h"
#include "eltwise_kernel_fs_bs_yx_bsv4_fsv32.h"
#include "eltwise_kernel_b_fs_yx_fsv4.h"
#include "eltwise_kernel_fs_b_yx_fsv32.h"
#include "eltwise_kernel_b_fs_yx_fsv16.h"
#include "eltwise_kernel_mixed_byxf_and_fs_b_yx_fsv32.h"

namespace kernel_selector {
eltwise_kernel_selector::eltwise_kernel_selector() {
    Attach<EltwiseKernelRef>();
    Attach<EltwiseKernel_vload8>();
    Attach<EltwiseKernel_fs_bs_yx_bsv4_fsv32>();
    Attach<EltwiseKernel_b_fs_yx_fsv4>();
    Attach<EltwiseKernel_fs_b_yx_fsv32>();
    Attach<EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32>();
    Attach<EltwiseKernel_b_fs_yx_fsv16>();
}

KernelsData eltwise_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::ELTWISE);
}
}  // namespace kernel_selector
