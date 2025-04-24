// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_b_fs_zyx_fsv16.h"

namespace kernel_selector {

class ConvolutionKernel_b_fs_zyx_fsv16_fp32 : public ConvolutionKernel_b_fs_zyx_fsv16 {
public:
    using Parent = ConvolutionKernel_b_fs_zyx_fsv16;

    ConvolutionKernel_b_fs_zyx_fsv16_fp32() : ConvolutionKernel_b_fs_zyx_fsv16(Datatype::F32) {}

    virtual ~ConvolutionKernel_b_fs_zyx_fsv16_fp32() {}
};
}  // namespace kernel_selector
