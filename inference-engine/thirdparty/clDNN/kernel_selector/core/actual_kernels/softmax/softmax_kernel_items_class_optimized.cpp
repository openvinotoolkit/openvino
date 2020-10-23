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

#include "softmax_kernel_items_class_optimized.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
// how many workitems we use to calculate item classes for one output, only 16 supported right now
static const auto workitems_per_classes = 16;

ParamsKey SoftmaxKerneItemsClassOptimized::GetSupportedKey() const { return GetDefaultSupportedKey(); }

SoftmaxKerneItemsClassOptimized::Parent::DispatchData SoftmaxKerneItemsClassOptimized::SetDefault(
    const softmax_params& params,
    const optional_params& optParams) const {
    auto dispatchData = Parent::SetDefault(params, optParams);

    auto& input = params.inputs[0];

    size_t item_class_count = 0;
    const auto global = GetSoftmaxDimGlobalSizes(params.dim, params.output);

    assert(global.size() == 3);

    switch (params.dim) {
        case SoftmaxDim::X:
            item_class_count = input.X().v;
            break;
        case SoftmaxDim::Y:
            item_class_count = input.Y().v;
            break;
        case SoftmaxDim::Z:
            item_class_count = input.Z().v;
            break;
        case SoftmaxDim::FEATURE:
            item_class_count = input.Feature().v;
            break;
        default:
            break;
    }

    dispatchData.gws[0] = global[0];
    dispatchData.gws[1] = global[1] * workitems_per_classes;  // we multiply it by workitems_per_classes because we split computations of
                                                         // one "full item classes output" into multiple workitems by "full item
                                                         // classes output" i mean N outputs where N is number of item classes.
    dispatchData.gws[2] = global[2];

    dispatchData.lws = { 1, static_cast<size_t>(workitems_per_classes), 1 };

    dispatchData.leftovers = item_class_count % workitems_per_classes;

    if (item_class_count >= 32) {
        dispatchData.efficiency = FORCE_PRIORITY_7;
    } else {
        dispatchData.efficiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    }

    return dispatchData;
}

JitConstants SoftmaxKerneItemsClassOptimized::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = SoftmaxItemsClassKernelBase::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("WORKITEMS_PER_CLASSES", workitems_per_classes));
    jit.AddConstant(MakeJitConstant("HAS_DRIVER_PROBLEMS", params.engineInfo.bIMADSupport));

    return jit;
}
KernelsData SoftmaxKerneItemsClassOptimized::GetKernelsData(const Params& params,
                                                            const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
