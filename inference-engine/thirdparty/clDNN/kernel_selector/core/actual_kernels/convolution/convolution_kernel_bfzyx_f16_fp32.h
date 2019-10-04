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

#include "convolution_kernel_bfzyx_f16.h"

namespace kernel_selector {

class ConvolutionKernel_bfzyx_f16_fp32 : public ConvolutionKernel_bfzyx_f16 {
public:
    using Parent = ConvolutionKernel_bfzyx_f16;

    ConvolutionKernel_bfzyx_f16_fp32() : ConvolutionKernel_bfzyx_f16(Datatype::F32) {}

    virtual ~ConvolutionKernel_bfzyx_f16_fp32() {}
};
}  // namespace kernel_selector
