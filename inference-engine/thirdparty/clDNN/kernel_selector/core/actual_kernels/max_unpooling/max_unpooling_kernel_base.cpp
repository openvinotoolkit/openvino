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


#include "max_unpooling_kernel_base.h"
#include <algorithm>

namespace kernel_selector {
bool MaxUnpoolingKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::MAX_UNPOOLING || o.GetType() != KernelType::MAX_UNPOOLING) {
        return false;
    }

    return true;
}

JitConstants MaxUnpoolingKernelBase::GetJitConstants(const max_unpooling_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    if (params.output.PitchesDifferFromLogicalDims())
        jit.AddConstant(MakeJitConstant("OUTPUT_PADDED", 1));
    return jit;
}

MaxUnpoolingKernelBase::DispatchData MaxUnpoolingKernelBase::SetDefault(const max_unpooling_params& params) const {
    const auto& input = params.inputs[0];

    DispatchData dispatchData;

    if (input.GetLayout() == DataLayout::bfyx || input.GetLayout() == DataLayout::byxf) {
        // Determine global work sizes.
        dispatchData.gws[2] = input.Batch().v * input.Feature().v;  // B, F
        dispatchData.gws[0] = Align(input.X().v, 32);               // X
        dispatchData.gws[1] = input.Y().v;                          // Y

        dispatchData.lws[0] = 32;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    } else {
        // Determine global work sizes.
        dispatchData.gws[0] = input.Batch().v * input.Feature().v;  // B, F
        dispatchData.gws[1] = input.X().v;                          // X
        dispatchData.gws[2] = input.Y().v;                          // Y

        dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
        while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
            --dispatchData.lws[0];
        }
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}

KernelsData MaxUnpoolingKernelBase::GetCommonKernelsData(const Params& params,
                                                         const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const max_unpooling_params& orgParams = static_cast<const max_unpooling_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<max_unpooling_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);
    kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});

    return {kd};
}
}  // namespace kernel_selector
