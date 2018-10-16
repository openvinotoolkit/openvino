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

#include "training_kernel_base.h"
#include "weight_bias_kernel_base.h"

namespace kernel_selector
{
    JitConstants training_kernel_base::GetJitConstants(const training_params& params) const
    {
        JitConstants jit = WeightBiasKernelBase::GetJitConstants(params);

        if (params.use_momentum)
        {
            jit.AddConstant(MakeJitConstant("MOMENTUM", 1));
            jit.AddConstant(MakeJitConstant("MOMENTUM_FACTOR", params.momentum_factor));
        }

        jit.AddConstant(MakeJitConstant("DECAY_RATE", params.weights_decay));

        return jit;
    }

}