/*
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
*/

#include "detection_output_kernel_ref.h"
#include "kernel_selector_utils.h"

#define PRIOR_BOX_SIZE 4  // Each prior-box consists of [xmin, ymin, xmax, ymax].

namespace kernel_selector {

ParamsKey DetectionOutputKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants DetectionOutputKernelRef::GetJitConstants(const detection_output_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& detectOutParams = params.detectOutParams;

    jit.AddConstants({
        MakeJitConstant("NUM_IMAGES", detectOutParams.num_images),
        MakeJitConstant("NUM_CLASSES", detectOutParams.num_classes),
        MakeJitConstant("KEEP_TOP_K", detectOutParams.keep_top_k),
        MakeJitConstant("TOP_K", detectOutParams.top_k),
        MakeJitConstant("BACKGROUND_LABEL_ID", detectOutParams.background_label_id),
        MakeJitConstant("CODE_TYPE", detectOutParams.code_type),
        MakeJitConstant("CONF_SIZE_X", detectOutParams.conf_size_x),
        MakeJitConstant("CONF_SIZE_Y", detectOutParams.conf_size_y),
        MakeJitConstant("CONF_PADDING_X", detectOutParams.conf_padding_x),
        MakeJitConstant("CONF_PADDING_Y", detectOutParams.conf_padding_y),
        MakeJitConstant("SHARE_LOCATION", detectOutParams.share_location),
        MakeJitConstant("VARIANCE_ENCODED_IN_TARGET", detectOutParams.variance_encoded_in_target),
        MakeJitConstant("NMS_THRESHOLD", detectOutParams.nms_threshold),
        MakeJitConstant("ETA", detectOutParams.eta),
        MakeJitConstant("CONFIDENCE_THRESHOLD", detectOutParams.confidence_threshold),
        MakeJitConstant("IMAGE_WIDTH", detectOutParams.input_width),
        MakeJitConstant("IMAGE_HEIGH", detectOutParams.input_heigh),
        MakeJitConstant("DECREASE_LABEL_ID", detectOutParams.decrease_label_id),
        MakeJitConstant("CLIP_BEFORE_NMS", detectOutParams.clip_before_nms),
        MakeJitConstant("CLIP_AFTER_NMS", detectOutParams.clip_after_nms),
        MakeJitConstant("ELEMENTS_PER_THREAD", detectOutParams.elements_per_thread),
        MakeJitConstant("PRIOR_COORD_OFFSET", detectOutParams.prior_coordinates_offset),
        MakeJitConstant("PRIOR_INFO_SIZE", detectOutParams.prior_info_size),
        MakeJitConstant("PRIOR_IS_NORMALIZED", detectOutParams.prior_is_normalized),
    });

    return jit;
}

int GetPartitionStep(int localWorkItemNum) {
    int step_size = 0;
    for (int temp = localWorkItemNum; temp > 1; temp /= 2) {
        step_size++;
    }
    return step_size;
}

size_t GetOptimalLocalClassSize(std::vector<size_t> gws, const EngineInfo& info) {
    const size_t lws_max = info.maxWorkGroupSize;
    const size_t optimal_lws_values[] = {8, 7, 6, 5, 4, 2, 1};
    size_t total_lws = gws[0] * gws[2];
    size_t localClassSize = 1;

    auto rest_lws = lws_max / total_lws;
    size_t lws_idx = 0;
    while (rest_lws < optimal_lws_values[lws_idx]) lws_idx++;
    while (gws[1] % optimal_lws_values[lws_idx]) lws_idx++;

    localClassSize = optimal_lws_values[lws_idx];
    total_lws *= optimal_lws_values[lws_idx];

    return localClassSize;
}

DetectionOutputKernelRef::DispatchData SetDefault(const detection_output_params& params, int idx) {
    DetectionOutputKernelRef::DispatchData dispatchData;
    const auto& input = params.inputs[0];
    const auto& detectOutParams = params.detectOutParams;
    constexpr size_t prior_box_size = 4;
    auto loc_feature_num = params.inputs[0].Feature().v;
    auto num_classes = detectOutParams.num_classes;
    auto num_loc_classes = (detectOutParams.share_location) ? 1 : num_classes;
    auto num_prior_boxes = (loc_feature_num / (num_loc_classes * prior_box_size));

    if (idx == 0) {
        if (detectOutParams.decrease_label_id) {
            dispatchData.gws = {input.Batch().v, num_prior_boxes, 1};
            dispatchData.lws = {1, 1, 1};
        } else {
            dispatchData.gws = {input.Batch().v, num_classes, 1};
            dispatchData.lws = {1, 1, 1};
        }
    } else if (idx == 1) {
        const size_t kSplitNum = 4;
        if (detectOutParams.decrease_label_id) {
            // dispatchData.gws = { 1, 1, 1};
            // dispatchData.lws = { 1, 1, 1};
            dispatchData.gws = {input.Batch().v, 1, kSplitNum};
            dispatchData.lws = {1, 1, kSplitNum};
        } else {
            // dispatchData.gws = { 1, 1, 1};
            // dispatchData.lws = { 1, 1, 1};
            dispatchData.gws = {input.Batch().v, num_classes, kSplitNum};
            const size_t kClassSize = GetOptimalLocalClassSize(dispatchData.gws, params.engineInfo);
            dispatchData.lws = {1, kClassSize, kSplitNum};
        }
    } else if (idx == 2) {
        if (detectOutParams.decrease_label_id) {
            dispatchData.gws = {input.Batch().v, 1, 1};
            // dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
            dispatchData.lws = {1, 1, 1};
        } else {
            dispatchData.gws = {input.Batch().v, num_classes, 1};
            // dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
            dispatchData.lws = {1, 1, 1};
        }
    } else {
        dispatchData.gws = {1, 1, 1};
        dispatchData.lws = {1, 1, 1};
    }
    // printf("idx[%d] gws: { %zd, %zd, %zd }\n", idx, dispatchData.gws[0], dispatchData.gws[1], dispatchData.gws[2]);
    // printf("idx[%d] lws: { %zd, %zd, %zd }\n", idx, dispatchData.lws[0], dispatchData.lws[1], dispatchData.lws[2]);

    return dispatchData;
}

void DetectionOutputKernelRef::SetKernelArguments(const detection_output_params& params, clKernelData& kernel, size_t idx) const {
    if (idx == 0) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});
    } else if (idx == 1) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});
    } else if (idx == 2) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});
    } else if (idx == 3) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});
    }
}

KernelsData DetectionOutputKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::DETECTION_OUTPUT && options.GetType() == KernelType::DETECTION_OUTPUT);

    constexpr size_t kKernelsNum = 4;
    KernelData kd = KernelData::Default<detection_output_params>(params, kKernelsNum);
    const detection_output_params& detectOutParams = static_cast<const detection_output_params&>(params);

    constexpr size_t prior_box_size = 4;
    auto num_of_images = detectOutParams.inputs[0].Batch().v;
    auto loc_feature_num = detectOutParams.inputs[0].Feature().v;
    auto num_classes = detectOutParams.detectOutParams.num_classes;
    auto num_loc_classes = (detectOutParams.detectOutParams.share_location) ? 1 : num_classes;
    auto num_prior_boxes = (loc_feature_num / (num_loc_classes * prior_box_size));

    constexpr size_t buffer_bytes = 16; // bboxes[xmin, ymin, xmax, ymax], scores[batchId, classId, boxId, score]
    size_t buffer_stride = num_prior_boxes * buffer_bytes;
    size_t buffer1_size = num_of_images * num_loc_classes * buffer_stride;
    size_t buffer2_size = num_of_images * num_classes * buffer_stride;
    size_t buffer3_size = num_of_images * num_classes * buffer_stride;
    size_t buffer4_size = num_of_images * num_classes * 4;
    // printf("GetKernelsData | buffer_stride = [%zd], buffer1_size = [%zd], buffer2/3_size = [%zd], buffer4_size = [%zd]\n",
    //        buffer_stride, buffer1_size, buffer2_size, buffer4_size);

    kd.internalBufferSizes.push_back(buffer1_size);
    kd.internalBufferSizes.push_back(buffer2_size);
    kd.internalBufferSizes.push_back(buffer3_size);
    kd.internalBufferSizes.push_back(buffer4_size);
    kd.internalBufferDataType = Datatype::F32;

    for (size_t i = 0; i < kKernelsNum; i++) {
        DispatchData dispatchData = SetDefault(detectOutParams, i);
        auto cldnnJit = GetJitConstants(detectOutParams);
        auto entryPoint = GetEntryPoint(kernelName, detectOutParams.layerID, options);
        cldnnJit.AddConstant(MakeJitConstant("BUFFER_STRIDE", buffer_stride));
        if (i == 0) {
            if (detectOutParams.detectOutParams.decrease_label_id) {
                //cldnnJit.AddConstant(MakeJitConstant("IS_ZERO_ITER_MXNET", "true"));
                cldnnJit.AddConstant(MakeJitConstant("IS_ZERO_ITER_MXNET_OPT", "true"));
            } else {
                // cldnnJit.AddConstant(MakeJitConstant("IS_ZERO_ITER_CAFFE", "true"));
                cldnnJit.AddConstant(MakeJitConstant("IS_ZERO_ITER_CAFFE_OPT", "true"));
            }
        } else if (i == 1) {
             if (detectOutParams.detectOutParams.decrease_label_id) {
                // cldnnJit.AddConstant(MakeJitConstant("IS_FIRST_ITER_MXNET", "true"));
                cldnnJit.AddConstants({MakeJitConstant("IS_FIRST_ITER_MXNET_OPT", "true"),
                                       MakeJitConstant("LOCAL_WORK_NUM", dispatchData.lws[2]),
                                       MakeJitConstant("PARTITION_STEP", GetPartitionStep(dispatchData.lws[2]))});
             } else {
                // cldnnJit.AddConstant(MakeJitConstant("IS_FIRST_ITER_CAFFE", "true"));
                cldnnJit.AddConstants({MakeJitConstant("IS_FIRST_ITER_CAFFE_OPT", "true"),
                                       MakeJitConstant("LOCAL_CLASS_NUM", dispatchData.lws[1]),
                                       MakeJitConstant("LOCAL_WORK_NUM", dispatchData.lws[2]),
                                       MakeJitConstant("PARTITION_STEP", GetPartitionStep(dispatchData.lws[2]))});
             }
        } else if (i == 2) {
            if (detectOutParams.detectOutParams.decrease_label_id) {
                //cldnnJit.AddConstant(MakeJitConstant("IS_SECOND_ITER_MXNET", "true"));
                cldnnJit.AddConstant(MakeJitConstant("IS_SECOND_ITER_MXNET_OPT", "true"));
            } else {
                //cldnnJit.AddConstant(MakeJitConstant("IS_SECOND_ITER_CAFFE", "true"));
                cldnnJit.AddConstant(MakeJitConstant("IS_SECOND_ITER_CAFFE_OPT", "true"));
            }
        } else {
            if (detectOutParams.detectOutParams.decrease_label_id) {
                cldnnJit.AddConstant(MakeJitConstant("IS_THIRD_ITER_MXNET", "true"));
            } else {
                cldnnJit.AddConstant(MakeJitConstant("IS_THIRD_ITER_CAFFE", "true"));
            }
        }

        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);
        auto& kernel = kd.kernels[i];
        KernelBase::CheckDispatchData(kernelName, dispatchData);
        kernel.workGroups.global = dispatchData.gws;
        kernel.workGroups.local  = dispatchData.lws;
        kernel.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
        SetKernelArguments(detectOutParams, kernel, i);
    }

    return {kd};
}

KernelsPriority DetectionOutputKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
