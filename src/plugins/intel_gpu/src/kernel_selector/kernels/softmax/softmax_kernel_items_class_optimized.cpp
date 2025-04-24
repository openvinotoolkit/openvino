// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_items_class_optimized.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
// how many workitems we use to calculate item classes for one output, only 16 supported right now
static const auto workitems_per_classes = 16;

inline static size_t get_class_pitch(const DataTensor& tensor, SoftmaxDim dim) {
    switch (dim) {
        case SoftmaxDim::X: return tensor.X().pitch;
        case SoftmaxDim::Y: return tensor.Y().pitch;
        case SoftmaxDim::Z: return tensor.Z().pitch;
        case SoftmaxDim::FEATURE: return tensor.Feature().pitch;
        case SoftmaxDim::BATCH: return tensor.Batch().pitch;
        default: return 0;
    }
}

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

ParamsKey SoftmaxKerneItemsClassOptimized::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::fb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::f);
    k.EnableOutputLayout(DataLayout::f);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableSoftmaxDim(SoftmaxDim::X);
    k.EnableSoftmaxDim(SoftmaxDim::Y);
    k.EnableSoftmaxDim(SoftmaxDim::Z);
    k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
    k.EnableSoftmaxDim(SoftmaxDim::BATCH);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey SoftmaxKerneItemsClassOptimized::get_required_device_features_key(const Params& params) const {
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

    dispatchData.dataSetsCount = dispatchData.gws[2];
    dispatchData.dataSetSize = GetItemClassCount(input, params.dim);
    dispatchData.leftovers = dispatchData.dataSetSize % workitems_per_classes;

    return dispatchData;
}

KernelsPriority SoftmaxKerneItemsClassOptimized::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const softmax_params&>(params);

    return GetItemClassCount(p.inputs[0], p.dim) >= 32 ? FORCE_PRIORITY_7 : DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants SoftmaxKerneItemsClassOptimized::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = SoftmaxItemsClassKernelBase::GetJitConstants(params, dispatchData);

    // sub_group_block_write requires
    // 1. aligned memory, therefore it can be utilized if memory is aligned by 16 bytes
    // 2. class dimension is innermost or all other dims equal to 1
    bool isSubGroupBlockIOEnabled = get_class_pitch(params.outputs[0], params.dim) == 1 &&
                                    get_class_pitch(params.inputs[0], params.dim) == 1 &&
                                    (dispatchData.dataSetSize * params.outputs[0].ElementSize()) % 16 == 0;

    jit.AddConstants({
        MakeJitConstant("LEFTOVERS", dispatchData.leftovers),
        MakeJitConstant("WORKITEMS_PER_CLASSES", workitems_per_classes),
        MakeJitConstant("HAS_DRIVER_PROBLEMS", params.engineInfo.supports_imad),
        MakeJitConstant("IS_SUBGROUP_BLOCK_IO_ENABLED", isSubGroupBlockIOEnabled),
    });

    return jit;
}
KernelsData SoftmaxKerneItemsClassOptimized::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}
}  // namespace kernel_selector
