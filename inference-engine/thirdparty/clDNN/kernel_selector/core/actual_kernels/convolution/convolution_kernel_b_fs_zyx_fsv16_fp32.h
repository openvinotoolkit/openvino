//
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
