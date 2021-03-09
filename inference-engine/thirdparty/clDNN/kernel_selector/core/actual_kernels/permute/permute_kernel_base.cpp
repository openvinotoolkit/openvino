// Copyright (c) 2021 Intel Corporation
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

#include "permute_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

bool PermuteKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::PERMUTE || o.GetType() != KernelType::PERMUTE) {
        return false;
    }
    const permute_params& params = static_cast<const permute_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op)) {
            return false;
        }
    }
    return true;
}

JitConstants PermuteKernelBase::GetJitConstants(const permute_params& params, const CommonDispatchData&) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    return jit;
}

KernelsData PermuteKernelBase::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);
    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 1, GetFusedPrimitiveInputsCount(params));

    return {kd};
}
}  // namespace kernel_selector
