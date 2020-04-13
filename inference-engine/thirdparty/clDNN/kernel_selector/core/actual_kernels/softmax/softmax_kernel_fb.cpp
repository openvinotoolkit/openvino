// Copyright (c) 2016-2020 Intel Corporation
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

#include "softmax_kernel_fb.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey SoftmaxKernel_fb::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableSoftmaxDim(SoftmaxDim::X);  // in case that it can be flatten
    k.EnableSoftmaxDim(SoftmaxDim::Y);
    k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
    k.EnableBatching();
    return k;
}

SoftmaxKernel_fb::Parent::DispatchData SoftmaxKernel_fb::SetDefault(const softmax_params& params,
                                                                    const optional_params& optParams) const {
    auto kd = Parent::SetDefault(params, optParams);
    // start with 1 thread per data set
    kd.gws0 = kd.dataSetsCount;
    kd.gws1 = 1;
    kd.itemsNum = kd.dataSetSize;

    kd.normIndex = 1;

    // We have two units of data per work item in current implementation.
    auto local_mem_per_wi = 2 * (kd.fp16UnitUsed ? sizeof(short) : sizeof(float));
    // Combining device execution and local memory restrictions to compute maximum possible LWS.
    auto max_lws = static_cast<std::size_t>(
        std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi));

    kd.lws0 = std::min(kd.dataSetsCount, max_lws);
    // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory
    // reads.
    while ((kd.itemsNum > 32 || kd.lws0 < kd.itemsNum) && (2 * kd.lws0 <= max_lws)) {
        kd.lws0 *= 2;
        kd.itemsNum /= 2;
    }

    kd.gws0 = kd.lws0;
    kd.gws1 = 1;
    kd.leftovers = (kd.dataSetSize * kd.dataSetsCount) % kd.lws0;

    assert(kd.itemsNum > 0 && kd.lws0 && kd.gws0 > 0);

    kd.efficiency = FORCE_PRIORITY_6;
    return kd;
}

bool kernel_selector::SoftmaxKernel_fb::Validate(const Params& params, const optional_params& o) const {
    if (!SoftmaxKernelBase::Validate(params, o)) {
        return false;
    }

    const auto& softmax_params = static_cast<const kernel_selector::softmax_params&>(params);

    auto kd = Parent::SetDefault(softmax_params, o);
    auto local_mem_per_wi = 2 * (kd.fp16UnitUsed ? sizeof(short) : sizeof(float));
    auto max_lws = static_cast<std::size_t>(
        std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi));

    size_t data_sets_count = softmax_params.inputs[0].Batch().v;
    if (data_sets_count > max_lws) {
        return false;
    }

    const auto& input = softmax_params.inputs[0];
    switch (softmax_params.dim) {
        case SoftmaxDim::X:
            return input.Y().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::Y:
            return input.X().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::FEATURE:
            return input.X().v == 1 && input.Y().v == 1;
        default:
            return false;
    }
}

KernelsData SoftmaxKernel_fb::GetKernelsData(const Params& params, const optional_params& optParams) const {
    if (!Validate(params, optParams)) {
        return {};
    }
    return GetCommonKernelsData(params, optParams);
}
}  // namespace kernel_selector