// Copyright (C) 2018-2022 Intel Corporation
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
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

KernelsPriority MulticlassNmsKernelRef::GetKernelsPriority(const Params&, const optional_params&) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool MulticlassNmsKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::MULTICLASS_NMS || o.GetType() != KernelType::MULTICLASS_NMS) {
        return false;
    }

    // FIXME opoluektov: more checks on the attribute values
    return true;
}

//int calculateSelectedBoxesStaticDimension(const multiclass_nms_params& params) {
//    int64_t max_output_boxes_per_class = 0;
//    if (params.nms_top_k >= 0)
//        max_output_boxes_per_class = std::min((int)param.num_boxes, param.nms_top_k);
//    else
//        max_output_boxes_per_class = param.num_boxes;
//
//    auto max_output_boxes_per_batch = max_output_boxes_per_class * param.num_classes;
//    if (param.keep_top_k >= 0)
//        max_output_boxes_per_batch = std::min<int>(max_output_boxes_per_batch, param.keep_top_k);
//
//    const auto dim = max_output_boxes_per_batch * param.num_batches;
//}

JitConstants MulticlassNmsKernelRef::GetJitConstants(const multiclass_nms_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    // FIXME opoluektov: hardcoding
    const auto num_batches = params.has_roisnum ? params.inputs[2].Batch().v : params.inputs[0].Batch().v;
    const auto num_classes = params.has_roisnum ? params.inputs[1].Batch().v : params.inputs[1].Feature().v;
    const auto num_boxes = params.inputs[0].Feature().v;

    int64_t max_output_boxes_per_class = 0;
    if (params.nms_top_k >= 0)
        max_output_boxes_per_class = std::min<int>(num_boxes, params.nms_top_k);
    else
        max_output_boxes_per_class = num_boxes;

    auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
    if (params.keep_top_k >= 0)
        max_output_boxes_per_batch = std::min<int>(max_output_boxes_per_batch, params.keep_top_k);

    const auto dim = max_output_boxes_per_batch * num_batches;

    jit.AddConstants({
        MakeJitConstant("SORT_RESULT_TYPE", params.sort_result_type),
        MakeJitConstant("SORT_RESULT_ACROSS_BATCH", params.sort_result_across_batch),
        MakeJitConstant("OUTPUT_INDICES_TYPE",
                        params.output_type == ov::element::i32 ? "int" : "long"),  // FIXME opoluektov
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

        MakeJitConstant("OUTPUT_DIM", dim),
    });

    if (params.has_roisnum) {
        jit.AddConstant(MakeJitConstant("HAS_ROISNUM", 1));
    }

    return jit;
}

using DispatchData = CommonDispatchData;

void MulticlassNmsKernelRef::PrepareKernelCommon(const multiclass_nms_params& params,
                                                 const optional_params& options,
                                                 std::vector<size_t> gws,
                                                 const std::string& stage_name,
                                                 size_t stage_index,
                                                 clKernelData& kernel) const {
    DispatchData dispatch_data;
    dispatch_data.gws = std::move(gws);
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);

    const auto entry_point = GetEntryPoint(kernelName, params.layerID, params, options, stage_index);
    auto cldnn_jit = GetJitConstants(params);
    cldnn_jit.AddConstant(MakeJitConstant(stage_name, "true"));

    const auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
    KernelBase::CheckDispatchData(kernelName, dispatch_data, params.engineInfo.maxWorkGroupSize);
    kernel.params.workGroups.global = dispatch_data.gws;
    kernel.params.workGroups.local = dispatch_data.lws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
}

void MulticlassNmsKernelRef::PrepareEverythingKernel(const multiclass_nms_params& params,
                                                     const optional_params& options,
                                                     clKernelData& kernel) const {
    //    const size_t roi_count = params.inputs[kScoresInputIdx].Batch().v;
    //    const size_t class_count = params.num_classes;

    PrepareKernelCommon(params, options, {1, 1, 1}, "MULTICLASS_STAGE_EVERYTHING", 0, kernel);

    for (auto i = 0; i < 2 + params.has_roisnum + 2; ++i) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, (uint32_t)i});
    }

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

    //    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kBoxesInputIdx});
    //    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kDeltasInputIdx});
    //    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kScoresInputIdx});
    //    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kImInfoInputIdx});
    //    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedBoxesBufferIdx});
    //    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedBoxAreasBufferIdx});
    //    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedScoresBufferIdx});
}

KernelsData MulticlassNmsKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    constexpr size_t kKernelCount = 1;  // FIXME opoluektov
    KernelData kd = KernelData::Default<multiclass_nms_params>(params, kKernelCount);
    const auto& op_params = static_cast<const multiclass_nms_params&>(params);

    //    const auto roi_count = op_params.inputs[kScoresInputIdx].Batch().v;
    //    const auto class_count = static_cast<size_t>(op_params.num_classes);

    // FIXME opoluektov: copy-paste
    const auto num_batches = op_params.has_roisnum ? op_params.inputs[2].Batch().v : op_params.inputs[0].Batch().v;
    const auto num_classes = op_params.has_roisnum ? op_params.inputs[1].Batch().v : op_params.inputs[1].Feature().v;
    const auto num_boxes = op_params.inputs[0].Feature().v;

    kd.internalBufferDataType = Datatype::F32;

    kd.internalBufferSizes.resize(1);
    kd.internalBufferSizes[0] =  (5*sizeof(double) + 3*sizeof(long)) * (num_batches * num_classes * num_boxes); // FIXME opoluektov more precice math plz

    //    kd.internalBufferSizes.resize(kBufferCount);
    //    kd.internalBufferSizes[kRefinedBoxesBufferIdx] = class_count * roi_count * 4 * sizeof(float);
    //    kd.internalBufferSizes[kRefinedBoxAreasBufferIdx] = class_count * roi_count * sizeof(float);
    //    kd.internalBufferSizes[kRefinedScoresBufferIdx] = class_count * roi_count * sizeof(float);
    //    kd.internalBufferSizes[kScoreClassIndexBufferIdx] = class_count * roi_count * 12;  // sizeof ScoreClassIndex
    //    kd.internalBufferSizes[kDetectionCountBufferIdx] = sizeof(uint32_t);

    PrepareEverythingKernel(op_params, options, kd.kernels[0]);

    return {kd};
}

}  // namespace kernel_selector
