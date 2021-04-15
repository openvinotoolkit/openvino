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

#include "non_max_suppression_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

Datatype NonMaxSuppressionKernelRef::GetAccumulatorType(const non_max_suppression_params& params) const {
    auto in_dt = params.inputs[0].GetDType();
    //auto in1_dt = params.inputs[1].GetDType();
    //auto in2_dt = params.inputs[2].GetDType();
    auto out_dt = params.output.GetDType();

    //printf("%ud, %ud\n", BytesPerElement(in1_dt), BytesPerElement(in2_dt));

    auto smaller_fp_type = [](const Datatype& current, const Datatype& candidate) -> Datatype {
        if (candidate != Datatype::F32 || candidate != Datatype::F16)
            return current;

        return BytesPerElement(candidate) < BytesPerElement(current) ? candidate : current;
    };

    Datatype fp_type = Datatype::F32;
    fp_type = smaller_fp_type(fp_type, in_dt);
    fp_type = smaller_fp_type(fp_type, out_dt);

    return fp_type;
}

ParamsKey NonMaxSuppressionKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants NonMaxSuppressionKernelRef::GetJitConstants(const non_max_suppression_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("SORT_RESULT_DESCENDING", params.sort_result_descending),
                      MakeJitConstant("BOX_ENCODING", params.box_encoding)});

    jit.AddConstant(MakeJitConstant("OUTPUT_NUM", params.output.Batch().v));

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    int32_t input_index = 2;
    if (params.has_num_select_per_class) {
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_IDX", input_index));
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_VAL", "convert_int(num_select_per_class[0])"));
        input_index++;
    } else {
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_VAL", 0));
    }

    if (params.has_iou_threshold) {
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_IDX", input_index));
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_VAL", "TO_ACCUMULATOR_TYPE(iou_threshold[0])"));
        input_index++;
    } else {
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_VAL", "ACCUMULATOR_VAL_ZERO"));
    }

    if (params.has_score_threshold) {
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_IDX", input_index));
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_VAL", "TO_ACCUMULATOR_TYPE(score_threshold[0])"));
        input_index++;
    } else {
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_VAL", "ACCUMULATOR_VAL_ZERO"));
    }

    if (params.has_soft_nms_sigma) {
        jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_IDX", input_index));
        jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_VAL", "TO_ACCUMULATOR_TYPE(soft_nms_sigma[0])"));
        input_index++;
    } else {
        jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_VAL", "ACCUMULATOR_VAL_ZERO"));
    }

    if (params.has_second_output) {
        jit.AddConstants({ MakeJitConstant("SELECTED_SCORES_TERM", true),
                           MakeJitConstant("SELECTED_SCORES", params.second_output) });
    }

    if (params.has_third_output) {
        jit.AddConstants({ MakeJitConstant("VALID_OUTPUTS_TERM", true),
                           MakeJitConstant("VALID_OUTPUTS", params.third_output) });
    }

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

NonMaxSuppressionKernelRef::DispatchData SetDefault(const non_max_suppression_params& params, int idx) {
    NonMaxSuppressionKernelRef::DispatchData dispatchData;

    const auto& input = params.inputs[1];
    if (idx == 0) {
        dispatchData.gws = {input.Batch().v, input.Feature().v, 256};
        dispatchData.lws = {1, 1, 256};
    } else if (idx == 1) {
        const size_t kSplitNum = 8;    // 1, 2, 4, 8
        dispatchData.gws = {input.Batch().v, input.Feature().v, kSplitNum};
        const size_t kClassSize = GetOptimalLocalClassSize(dispatchData.gws, params.engineInfo);
        dispatchData.lws = {1, kClassSize, kSplitNum};
    } else if (idx == 2) {
        dispatchData.gws = {input.Batch().v, input.Feature().v, 1};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    } else {
        dispatchData.gws = {1, 1, 1};
        dispatchData.lws = {1, 1, 1};
    }

    return dispatchData;
}

bool NonMaxSuppressionKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::NON_MAX_SUPPRESSION || o.GetType() != KernelType::NON_MAX_SUPPRESSION) {
        return false;
    }

    const non_max_suppression_params& params = static_cast<const non_max_suppression_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

/*
 *  INPUT[0]: boxes
 *  INPUT[1]: scores
 *  INTERNAL_BUFFER[0]: intermidiate_sorted_box
 *  INTERNAL_BUFFER[1]: intermidiate_selected_box
 *  INTERNAL_BUFFER[2]: intermidiate_out_sorted_box
 *  INTERNAL_BUFFER[3]: intermidiate_sorted_box_num
 *
 */
void NonMaxSuppressionKernelRef::SetKernelArguments(const non_max_suppression_params& params,
                                                    clKernelData& kernel, size_t idx) const {
    uint32_t num_select_per_class_idx = 0;
    uint32_t iou_threshold_idx = 0;
    uint32_t score_threshold_idx = 0;
    uint32_t soft_nms_sigma_idx = 0;

    uint32_t input_idx = 2;
    if (params.has_num_select_per_class) {
        num_select_per_class_idx = input_idx++;
    }
    if (params.has_iou_threshold) {
        iou_threshold_idx = input_idx++;
    }
    if (params.has_score_threshold) {
        score_threshold_idx = input_idx++;
    }
    if (params.has_soft_nms_sigma) {
        soft_nms_sigma_idx = input_idx++;
    }


    if (idx == 0) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 3 });

        if (score_threshold_idx > 0)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, score_threshold_idx });
    } else if (idx == 1) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 3 });
    } else if (idx == 2) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 3 });

        if (num_select_per_class_idx > 0)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, num_select_per_class_idx });
        if (iou_threshold_idx > 0)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, iou_threshold_idx });
        if (score_threshold_idx > 0)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, score_threshold_idx });
        if (soft_nms_sigma_idx > 0)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, soft_nms_sigma_idx });
    } else if (idx == 3) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2 });

        if (params.has_second_output)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::SECOND_OUTPUT, 0 });
        if (params.has_third_output)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::THIRD_OUTPUT, 0 });
    }
}

KernelsData NonMaxSuppressionKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    constexpr size_t kKernelsNum = 4;
    KernelData kd = KernelData::Default<non_max_suppression_params>(params, kKernelsNum);
    const non_max_suppression_params& orgParams = static_cast<const non_max_suppression_params&>(params);

    // Assign internel buffer
    constexpr size_t intermidiate_bytes = 20;
    auto batch_num = orgParams.inputs[1].Batch().v;
    auto class_num = orgParams.inputs[1].Feature().v;
    auto boxes_num = orgParams.inputs[0].Feature().v;
    size_t buffer_stride = boxes_num * intermidiate_bytes;
    size_t buffer_size = batch_num * class_num * buffer_stride;
    size_t sel_num_buffer_size = batch_num * class_num * 4;

    kd.internalBufferSizes.push_back(buffer_size);
    kd.internalBufferSizes.push_back(buffer_size);
    kd.internalBufferSizes.push_back(buffer_size);
    kd.internalBufferSizes.push_back(sel_num_buffer_size);
    kd.internalBufferDataType = Datatype::F32;

    // Build clKernelData.
    for (size_t i = 0; i < kKernelsNum; i++) {
        DispatchData dispatchData = SetDefault(orgParams, i);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto cldnn_jit = GetJitConstants(orgParams);
        cldnn_jit.AddConstant(MakeJitConstant("BUFFER_STRIDE", buffer_stride));

        if (i == 0) {
            size_t num_bit_mask = CeilDiv(boxes_num, 8);
            size_t num_score_per_item = RoundUp(CeilDiv(boxes_num, 256), 8);
            size_t num_score_block = CeilDiv(boxes_num, num_score_per_item);
            // printf("num_score_per_item: %zd = RoundUp((%zd - 1)/256 + 1, 8), num_score_block: %zd, num_bit_mask: %zd\n"
            //         , num_score_per_item, boxes_num, num_score_block, num_bit_mask);
            cldnn_jit.AddConstants({ MakeJitConstant("NUM_BIT_MASK", num_bit_mask)
                                   , MakeJitConstant("NUM_SCORE_PER_ITEM", num_score_per_item)
                                   , MakeJitConstant("NUM_SCORE_BLOCK", num_score_block)
                                   , MakeJitConstant("IS_ZERO_ITER", "true")});
        } else if (i == 1) {
            cldnn_jit.AddConstants({ MakeJitConstant("IS_FIRST_ITER", "true")
                                   , MakeJitConstant("LOCAL_CLASS_NUM", dispatchData.lws[1])
                                   , MakeJitConstant("LOCAL_WORK_NUM", dispatchData.lws[2])
                                   , MakeJitConstant("PARTITION_STEP", GetPartitionStep(dispatchData.lws[2]))});
        } else if (i == 2) {
            cldnn_jit.AddConstant(MakeJitConstant("IS_SECOND_ITER", "true"));
        } else {
            cldnn_jit.AddConstant(MakeJitConstant("IS_THIRD_ITER", "true"));
        }

        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[i];
        KernelBase::CheckDispatchData(kernelName, dispatchData);
        kernel.workGroups.global = dispatchData.gws;
        kernel.workGroups.local  = dispatchData.lws;
        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
        SetKernelArguments(orgParams, kernel, i);
    }

    return {kd};
}

KernelsPriority NonMaxSuppressionKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
