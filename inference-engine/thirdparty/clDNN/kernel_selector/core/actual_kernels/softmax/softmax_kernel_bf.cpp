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

#include "softmax_kernel_bf.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey SoftmaxKernel_bf::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableSoftmaxDim(SoftmaxDim::X);  // in case that it can be flatten
    k.EnableSoftmaxDim(SoftmaxDim::Y);
    k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
    k.EnableBatching();
    return k;
}

SoftmaxKernel_bf::Parent::DispatchData SoftmaxKernel_bf::SetDefault(const softmax_params& params,
                                                                    const optional_params& optParams) const {
    auto kd = Parent::SetDefault(params, optParams);

    // start with 1 thread per data set
    kd.gws0 = 1;
    kd.gws1 = kd.dataSetsCount;
    kd.itemsNum = kd.dataSetSize;

    kd.normIndex = 0;

    // We have two units of data per work item in current implementation.
    auto local_mem_per_wi = 2 * (kd.fp16UnitUsed ? sizeof(short) : sizeof(float));
    // Combining device execution and local memory restrictions to compute maximum possible LWS.
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);

    kd.lws0 = 1;
    // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory
    // reads.
    while ((kd.itemsNum > 32 || kd.lws0 < kd.itemsNum) && (2 * kd.lws0 <= max_lws)) {
        kd.lws0 *= 2;
        kd.itemsNum /= 2;
    }

    assert((kd.itemsNum + 1) * kd.lws0 >= kd.dataSetSize && "More than 'lws0' items per batch remains! Lws too small?");

    kd.gws0 = kd.lws0;
    kd.leftovers = kd.dataSetSize % kd.lws0;

    assert(kd.itemsNum > 0 && kd.lws0 && kd.gws0 > 0);

    kd.efficiency = FORCE_PRIORITY_6;
    return kd;
}

KernelsData SoftmaxKernel_bf::GetKernelsData(const Params& params, const optional_params& optionalParams) const {
    return GetCommonKernelsData(params, optionalParams);
}
}  // namespace kernel_selector