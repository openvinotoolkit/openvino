/*
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
*/

#include "binary_convolution_kernel_selector.h"
#include "binary_convolution_kernel_ref.h"
#include "binary_convolution_kernel_generic.h"
#include "binary_convolution_kernel_1x1.h"
#include "binary_convolution_kernel_1x1_b_fs_yx_fsv16.h"

namespace kernel_selector {
binary_convolution_kernel_selector::binary_convolution_kernel_selector() {
    Attach<BinaryConvolutionKernel1x1>();
    Attach<BinaryConvolutionKernel1x1_b_fs_yx_fsv16>();
    Attach<BinaryConvolutionKernelGeneric>();
    Attach<BinaryConvolutionKernelRef>();
}

KernelsData binary_convolution_kernel_selector::GetBestKernels(const Params& params,
                                                               const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::BINARY_CONVOLUTION);
}
}  // namespace kernel_selector
