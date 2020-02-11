// Copyright (c) 2016 Intel Corporation
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

#include "softmax_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey SoftmaxKernelRef::GetSupportedKey() const { return GetDefaultSupportedKey(); }

SoftmaxKernelRef::Parent::DispatchData SoftmaxKernelRef::SetDefault(const softmax_params& params,
                                                                    const optional_params& optParams) const {
    auto runInfo = Parent::SetDefault(params, optParams);

    const auto global = GetSoftmaxDimGlobalSizes(params.dim, params.output);

    assert(global.size() == 3);

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    runInfo.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return runInfo;
}

KernelsData SoftmaxKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
