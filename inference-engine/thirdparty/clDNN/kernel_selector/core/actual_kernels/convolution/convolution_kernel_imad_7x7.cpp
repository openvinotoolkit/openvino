/*
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
*/

#include "convolution_kernel_imad_7x7.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector {

    JitConstants
        ConvolutionKernel_imad_7x7::GetJitConstants(
            const convolution_params& params,
            const DispatchData&       kd) const
    {
        auto mem_consts = Parent::GetJitConstants(params, kd);

        mem_consts.AddConstants({
            // Block reading optimization is implemented for 3x3 only.
            // For 7x7 it should be disabled.
            MakeJitConstant("NON_BLOCK_LOAD", 1),
        });
        return mem_consts;
    }
}
