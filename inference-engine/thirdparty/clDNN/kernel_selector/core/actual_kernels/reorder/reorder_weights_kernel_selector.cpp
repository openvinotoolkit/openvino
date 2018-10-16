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

#include "reorder_weights_kernel_selector.h"
#include "reorder_weights_kernel.h"
#include "reorder_weights_winograd_2x3_kernel.h"
#include "reorder_weights_winograd_6x3_kernel.h"
#include "reorder_weights_image_fyx_b_kernel.h"
#include "reorder_weights_image_winograd_6x3_kernel.h"
 
namespace kernel_selector {

    ReorderWeightsKernelSelctor::ReorderWeightsKernelSelctor()
    {
        Attach<ReorderWeightsKernel>();
        Attach<ReorderWeightsWinograd2x3Kernel>();
        Attach<ReorderWeightsWinograd6x3Kernel>();
        Attach<ReorderWeightsImage_fyx_b_Kernel>();
        Attach<ReorderWeightsImageWinograd6x3Kernel>();
    }

    KernelsData ReorderWeightsKernelSelctor::GetBestKernels(const Params& params, const optional_params& options) const
    {
        return GetNaiveBestKernel(params, options, KernelType::REORDER);
    }
}