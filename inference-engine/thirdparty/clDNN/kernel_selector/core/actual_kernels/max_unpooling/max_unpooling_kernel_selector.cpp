// Copyright (c) 2018 Intel Corporation
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


#include "max_unpooling_kernel_selector.h"
#include "max_unpooling_kernel_gpu_ref.h"

namespace kernel_selector {

max_unpooling_kernel_selector::max_unpooling_kernel_selector() { Attach<MaxUnpoolingKernelGPURef>(); }

KernelsData max_unpooling_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::MAX_UNPOOLING);
}
}  // namespace kernel_selector