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

#include "softmax_kernel_selector.h"
#include "softmax_kernel_ref.h"
#include "softmax_kernel_bf.h"
#include "softmax_kernel_fb.h"
#include "softmax_kernel_items_class_optimized.h"

namespace kernel_selector {

    softmax_kernel_selector::softmax_kernel_selector()
    {
        Attach<SoftmaxKernelRef>();
        Attach<SoftmaxKernel_bf>();
        Attach<SoftmaxKernel_fb>();
        Attach<SoftmaxKerneItemsClassOptimized>();
    }

    KernelsData softmax_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const
    {
        return GetNaiveBestKernel(params, options, KernelType::SOFT_MAX);
    }
}