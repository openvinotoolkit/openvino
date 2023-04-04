// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_items_class_optimized.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
// how many workitems we use to calculate item classes for one output, only 16 supported right now
static const auto workitems_per_classes = 16;

inline static size_t GetItemClassCount(const DataTensor& input, SoftmaxDim dim) {
    size_t item_class_count = 0;

    switch (dim) {
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
        case SoftmaxDim::BATCH:
            item_class_count = input.Batch().v;
            break;
        default:
            break;
    }

    return item_class_count;
}

ParamsKey SoftmaxKerneItemsClassOptimized::GetSupportedKey() const { return GetDefaultSupportedKey(); }

DeviceFeaturesKey SoftmaxKerneItemsClassOptimized::get_required_device_features_key(const Params& params, const optional_params& options) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_reduce();
    k.requires_reqd_subgroup_size();

    return k;
}

SoftmaxKerneItemsClassOptimized::Parent::DispatchData SoftmaxKerneItemsClassOptimized::SetDefault(const softmax_params& params) const {
    auto dispatchData = Parent::SetDefault(params);

    auto& input = params.inputs[0];

    const auto global = GetSoftmaxDimGlobalSizes(params.dim, params.outputs[0]);

    assert(global.size() == 3);

    dispatchData.gws[0] = global[0];
    dispatchData.gws[1] = global[1] * workitems_per_classes;  // we multiply it by workitems_per_classes because we split computations of
                                                              // one "full item classes output" into multiple workitems by "full item
                                                              // classes output" i mean N outputs where N is number of item classes.
    dispatchData.gws[2] = global[2];

    dispatchData.lws = { 1, static_cast<size_t>(workitems_per_classes), 1 };

    dispatchData.leftovers = GetItemClassCount(input, params.dim) % workitems_per_classes;

    return dispatchData;
}

KernelsPriority SoftmaxKerneItemsClassOptimized::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    const auto& p = static_cast<const softmax_params&>(params);

    return GetItemClassCount(p.inputs[0], p.dim) >= 32 ? FORCE_PRIORITY_7 : DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants SoftmaxKerneItemsClassOptimized::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = SoftmaxItemsClassKernelBase::GetJitConstants(params, dispatchData);

    jit.AddConstants({
        MakeJitConstant("LEFTOVERS", dispatchData.leftovers),
        MakeJitConstant("WORKITEMS_PER_CLASSES", workitems_per_classes),
        MakeJitConstant("HAS_DRIVER_PROBLEMS", params.engineInfo.supports_imad),
    });

    return jit;
}
KernelsData SoftmaxKerneItemsClassOptimized::GetKernelsData(const Params& params,
                                                            const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
