// Copyright (c) 2018-2020 Intel Corporation
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

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// broadcast_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct broadcast_params : public base_params {
    broadcast_params() : base_params(KernelType::BROADCAST) {}
    std::vector<uint16_t> input_order;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// broadcast_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct broadcast_optional_params : optional_params {
    broadcast_optional_params() : optional_params(KernelType::BROADCAST) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BroadcastKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BroadcastKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const broadcast_params& params) const;
    static DispatchData SetDefault(const broadcast_params& params);
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
