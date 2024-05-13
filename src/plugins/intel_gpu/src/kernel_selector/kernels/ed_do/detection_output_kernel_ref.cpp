// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output_kernel_ref.h"

#include <algorithm>
#include <string>

#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey ExperimentalDetectronDetectionOutputKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);

    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableTensorPitches();
    return k;
}

KernelsPriority ExperimentalDetectronDetectionOutputKernelRef::GetKernelsPriority(const Params&) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool ExperimentalDetectronDetectionOutputKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT) {
        return false;
    }
    return true;
}

constexpr int kBoxesInputIdx = 0;
constexpr int kDeltasInputIdx = 1;
constexpr int kScoresInputIdx = 2;
constexpr int kImInfoInputIdx = 3;
constexpr int kOutputClassesInputIdx = 1;
constexpr int kOutputScoresInputIdx = 2;

constexpr int kRefinedBoxesBufferIdx = 0;
constexpr int kRefinedBoxAreasBufferIdx = 1;
constexpr int kRefinedScoresBufferIdx = 2;
constexpr int kScoreClassIndexBufferIdx = 3;
constexpr int kDetectionCountBufferIdx = 4;

constexpr int kBufferCount = 5;

constexpr int kOutputIdx = 0;

JitConstants ExperimentalDetectronDetectionOutputKernelRef::GetJitConstants(
    const experimental_detectron_detection_output_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("SCORE_THRESHOLD", params.score_threshold),
        MakeJitConstant("NMS_THRESHOLD", params.nms_threshold),
        MakeJitConstant("NUM_CLASSES", params.num_classes),
        MakeJitConstant("POST_NMS_COUNT", params.post_nms_count),
        MakeJitConstant("MAX_DETECTIONS_PER_IMAGE", params.max_detections_per_image),
        MakeJitConstant("MAX_DELTA_LOG_WH", params.max_delta_log_wh),
        MakeJitConstant("DELTA_WEIGHT_X", params.deltas_weights[0]),
        MakeJitConstant("DELTA_WEIGHT_Y", params.deltas_weights[1]),
        MakeJitConstant("DELTA_WEIGHT_LOG_W", params.deltas_weights[2]),
        MakeJitConstant("DELTA_WEIGHT_LOG_H", params.deltas_weights[3]),

        MakeJitConstant("ROI_COUNT", params.inputs[kScoresInputIdx].Batch().v),
    });

    if (params.class_agnostic_box_regression) {
        jit.AddConstant(MakeJitConstant("CLASS_AGNOSTIC_BOX_REGRESSION", true));
    }
    if (!SimpleLayout(params.inputs[0].GetLayout())) {
        jit.AddConstant(MakeJitConstant("USE_BLOCKED_FORMAT", true));
    }
    return jit;
}

using DispatchData = CommonDispatchData;

void ExperimentalDetectronDetectionOutputKernelRef::PrepareKernelCommon(
    const experimental_detectron_detection_output_params& params,
    std::vector<size_t> gws,
    const std::string& stage_name,
    size_t stage_index,
    clKernelData& kernel) const {
    DispatchData dispatch_data;
    dispatch_data.gws = std::move(gws);
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);

    const auto entry_point = GetEntryPoint(kernelName, params.layerID, params, stage_index);
    auto cldnn_jit = GetJitConstants(params);
    cldnn_jit.AddConstant(MakeJitConstant(stage_name, "true"));

    const auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
    KernelBase::CheckDispatchData(kernelName, dispatch_data, params.engineInfo.maxWorkGroupSize);
    kernel.params.workGroups.global = dispatch_data.gws;
    kernel.params.workGroups.local = dispatch_data.lws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
}

void ExperimentalDetectronDetectionOutputKernelRef::PrepareRefineBoxesKernel(
    const experimental_detectron_detection_output_params& params,
    clKernelData& kernel) const {
    const size_t roi_count = params.inputs[kScoresInputIdx].Batch().v;
    const size_t class_count = params.class_agnostic_box_regression ? params.num_classes - 1 : params.num_classes;

    PrepareKernelCommon(params, {roi_count, class_count, 1}, "EDDO_STAGE_0_REFINE_BOXES", 0, kernel);

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kBoxesInputIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kDeltasInputIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kScoresInputIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kImInfoInputIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedBoxesBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedBoxAreasBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedScoresBufferIdx});
}

void ExperimentalDetectronDetectionOutputKernelRef::PrepareNmsClassWiseKernel(
    const experimental_detectron_detection_output_params& params,
    clKernelData& kernel) const {
    PrepareKernelCommon(params, {1, 1, 1}, "EDDO_STAGE_1_NMS", 1, kernel);

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedScoresBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedBoxesBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedBoxAreasBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kScoreClassIndexBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kDetectionCountBufferIdx});
}

void ExperimentalDetectronDetectionOutputKernelRef::PrepareTopKDetectionsKernel(
    const experimental_detectron_detection_output_params& params,
    clKernelData& kernel) const {
    PrepareKernelCommon(params, {1, 1, 1}, "EDDO_STAGE_2_TOPK", 2, kernel);

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kScoreClassIndexBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kDetectionCountBufferIdx});
}

void ExperimentalDetectronDetectionOutputKernelRef::PrepareCopyOutputKernel(
    const experimental_detectron_detection_output_params& params,
    clKernelData& kernel) const {
    PrepareKernelCommon(params,
                        {static_cast<size_t>(params.max_detections_per_image), 1, 1},
                        "EDDO_STAGE_3_COPY_OUTPUT",
                        3,
                        kernel);

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kScoreClassIndexBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kDetectionCountBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, kRefinedBoxesBufferIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, kOutputIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, kOutputClassesInputIdx});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, kOutputScoresInputIdx});
}

KernelsData ExperimentalDetectronDetectionOutputKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    constexpr size_t kKernelCount = 4;
    KernelData kd = KernelData::Default<experimental_detectron_detection_output_params>(params, kKernelCount);
    const auto& eddo_params = static_cast<const experimental_detectron_detection_output_params&>(params);

    const auto roi_count = eddo_params.inputs[kScoresInputIdx].Batch().v;
    const auto class_count = static_cast<size_t>(eddo_params.num_classes);

    kd.internalBufferDataType = Datatype::F32;

    kd.internalBufferSizes.resize(kBufferCount);
    kd.internalBufferSizes[kRefinedBoxesBufferIdx] = class_count * roi_count * 4 * sizeof(float);
    kd.internalBufferSizes[kRefinedBoxAreasBufferIdx] = class_count * roi_count * sizeof(float);
    kd.internalBufferSizes[kRefinedScoresBufferIdx] = class_count * roi_count * sizeof(float);
    kd.internalBufferSizes[kScoreClassIndexBufferIdx] = class_count * roi_count * 12;  // sizeof ScoreClassIndex
    kd.internalBufferSizes[kDetectionCountBufferIdx] = sizeof(uint32_t);

    PrepareRefineBoxesKernel(eddo_params, kd.kernels[0]);
    PrepareNmsClassWiseKernel(eddo_params, kd.kernels[1]);
    PrepareTopKDetectionsKernel(eddo_params, kd.kernels[2]);
    PrepareCopyOutputKernel(eddo_params, kd.kernels[3]);

    return {kd};
}

}  // namespace kernel_selector
