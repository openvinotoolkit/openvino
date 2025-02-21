// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiclass_nms_kernel_ref.h"

#include <algorithm>
#include <string>

#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey MulticlassNmsKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();

    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableTensorPitches();

    return k;
}

KernelsPriority MulticlassNmsKernelRef::GetKernelsPriority(const Params&) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool MulticlassNmsKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::MULTICLASS_NMS) {
        return false;
    }

    return true;
}

namespace {
MulticlassNmsKernelRef::DispatchData SetDefault(const multiclass_nms_params& params, size_t stage_idx) {
    MulticlassNmsKernelRef::DispatchData dispatch_data;

    enum KernelStages : size_t {
        Main = 0,
        SortAcrossBatches = 1,
        FillOutputs = 2
    };

    if (stage_idx == KernelStages::Main || stage_idx == KernelStages::FillOutputs) {
        const auto num_batches = params.has_roisnum ? params.inputs[2].Batch().v : params.inputs[1].Batch().v;
        dispatch_data.gws = {num_batches, 1, 1};
    } else {
        dispatch_data.gws = {1, 1, 1};
    }
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);

    return dispatch_data;
}
}  // namespace

void MulticlassNmsKernelRef::SetKernelArguments(const multiclass_nms_params& params,
                                                size_t idx, cldnn::arguments_desc& arguments) const {
    for (auto i = 0; i < (params.has_roisnum ? 3 : 2); ++i) {
        arguments.push_back({ArgumentDescriptor::Types::INPUT, (uint32_t)i});
    }
    arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
    arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 2});
}

JitConstants MulticlassNmsKernelRef::GetJitConstants(const multiclass_nms_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto num_batches = params.has_roisnum ? params.inputs[2].Batch().v : params.inputs[1].Batch().v;
    const int64_t num_classes = params.has_roisnum ? params.inputs[0].Batch().v : params.inputs[1].Feature().v;
    const auto num_boxes = params.inputs[0].Feature().v;

    auto real_num_classes = num_classes;
    if (params.background_class >= 0 && params.background_class < num_classes) {
        real_num_classes = std::max<int64_t>(1ll, num_classes - 1);
    }

    int64_t max_output_boxes_per_class = 0;
    if (params.nms_top_k >= 0) {
        max_output_boxes_per_class = std::min<int>(static_cast<int>(num_boxes), params.nms_top_k);
    } else {
        max_output_boxes_per_class = num_boxes;
    }

    auto max_output_boxes_per_batch = max_output_boxes_per_class * real_num_classes;
    if (params.keep_top_k >= 0)
        max_output_boxes_per_batch = std::min<int>(max_output_boxes_per_batch, params.keep_top_k);

    jit.AddConstants({
        MakeJitConstant("SORT_RESULT_TYPE", static_cast<int>(params.sort_result_type)),
        MakeJitConstant("SORT_RESULT_ACROSS_BATCH", params.sort_result_across_batch),
        MakeJitConstant("IOU_THRESHOLD", params.iou_threshold),
        MakeJitConstant("SCORE_THRESHOLD", params.score_threshold),
        MakeJitConstant("NMS_TOP_K", params.nms_top_k),
        MakeJitConstant("KEEP_TOP_K", params.keep_top_k),
        MakeJitConstant("BACKGROUND_CLASS", params.background_class),
        MakeJitConstant("NORMALIZED", params.normalized),
        MakeJitConstant("NMS_ETA", params.nms_eta),

        MakeJitConstant("NUM_BOXES", num_boxes),
        MakeJitConstant("NUM_CLASSES", num_classes),
        MakeJitConstant("NUM_BATCHES", num_batches),

        MakeJitConstant("MAX_OUTPUT_BOXES_PER_BATCH", max_output_boxes_per_batch),
    });

    if (params.has_roisnum) {
        jit.AddConstant(MakeJitConstant("HAS_ROISNUM", 1));
    }

    return jit;
}

KernelsData MulticlassNmsKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    constexpr size_t kKernelsNum = 3;
    KernelData kd = KernelData::Default<multiclass_nms_params>(params, kKernelsNum);
    const auto& op_params = static_cast<const multiclass_nms_params&>(params);

    const auto num_batches = op_params.has_roisnum ? op_params.inputs[2].Batch().v : op_params.inputs[0].Batch().v;
    const auto num_classes = op_params.has_roisnum ? op_params.inputs[1].Batch().v : op_params.inputs[1].Feature().v;
    const auto num_boxes = op_params.inputs[0].Feature().v;

    // buffer for BoxInfos
    kd.internalBufferDataType = Datatype::F32;
    kd.internalBuffers.resize(1);
    // double: 4 coordinates + 1 score; long: 1 class_idx + 1 batch_idx + 1 index
    const auto box_size = (4 + 1) * sizeof(double) + 3 * sizeof(long);
    const auto total_boxes = num_batches * num_classes * num_boxes;
    kd.internalBuffers[0] = box_size * total_boxes;
    const auto common_jit_constants = GetJitConstants(op_params);

    for (size_t i = 0; i < kKernelsNum; ++i) {
        const auto dispatch_data = SetDefault(op_params, i);
        const auto entry_point = GetEntryPoint(kernelName, op_params.layerID, params, i);
        auto cldnn_jit = common_jit_constants;
        cldnn_jit.AddConstant(MakeJitConstant("MULTICLASSNMS_STAGE_" + std::to_string(i), "true"));

        const auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
        KernelBase::CheckDispatchData(kernelName, dispatch_data, params.engineInfo.maxWorkGroupSize);
        auto& kernel = kd.kernels[i];

        kernel.params.workGroups.global = dispatch_data.gws;
        kernel.params.workGroups.local = dispatch_data.lws;
        kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
        SetKernelArguments(op_params, i, kernel.params.arguments);
    }

    return {kd};
}
}  // namespace kernel_selector
