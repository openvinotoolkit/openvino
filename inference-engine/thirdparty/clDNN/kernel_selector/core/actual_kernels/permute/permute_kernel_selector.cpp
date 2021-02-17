// Copyright (c) 2021 Intel Corporation
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


#include "permute_kernel_selector.h"
#include "permute_kernel_ref.h"
#include "permute_kernel_tile_8x8_4x4.h"

namespace kernel_selector {

permute_kernel_selector::permute_kernel_selector() {
    Attach<PermuteKernelRef>();
    Attach<PermuteKernel_tile_8x8_4x4>();
}

KernelsData permute_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::PERMUTE);
}
}  // namespace kernel_selector
