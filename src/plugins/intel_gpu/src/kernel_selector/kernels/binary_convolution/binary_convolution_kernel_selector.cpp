// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
