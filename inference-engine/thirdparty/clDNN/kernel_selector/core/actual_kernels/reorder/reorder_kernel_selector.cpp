/*
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
*/

#include "reorder_kernel_selector.h"
#include "reorder_kernel.h"
#include "reorder_kernel_fast_b1.h"
#include "reorder_from_winograd_2x3_kernel.h"
#include "reorder_to_winograd_2x3_kernel.h"
#include "reorder_kernel_to_yxfb_batched.h"

namespace kernel_selector {

    reorder_kernel_selector::reorder_kernel_selector()
    {
        Attach<ReorderKernelRef>();
        Attach<ReorderKernelFastBatch1>();
        Attach<ReorderFromWinograd2x3Kernel>();
        Attach<ReorderToWinograd2x3Kernel>();
        Attach<ReorderKernel_to_yxfb_batched>();
    }

    KernelsData reorder_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const
    {
        return GetNaiveBestKernel(params, options, KernelType::REORDER);
    }
}