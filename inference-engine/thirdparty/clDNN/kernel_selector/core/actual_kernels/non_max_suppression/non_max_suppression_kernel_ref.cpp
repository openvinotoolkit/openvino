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

    if (params.has_num_select_per_class) {
        char inputTypeStr[128];
        snprintf(inputTypeStr, sizeof(inputTypeStr), "INPUT%d_TYPE", params.GetIndexNumSelectPerClass());
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_TYPE", std::string(inputTypeStr)));
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_VAL", "convert_int(num_select_per_class[0])"));
    } else {
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_VAL", 0));
    }

    if (params.has_iou_threshold) {
        char inputTypeStr[128];
        snprintf(inputTypeStr, sizeof(inputTypeStr), "INPUT%d_TYPE", params.GetIndexIouThreshold());
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_TYPE", std::string(inputTypeStr)));
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_VAL", "convert_float(iou_threshold[0])"));
    } else {
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_VAL", "0.0f"));
    }

    if (params.has_score_threshold) {
        char inputTypeStr[128];
        snprintf(inputTypeStr, sizeof(inputTypeStr), "INPUT%d_TYPE", params.GetIndexScoreThreshold());
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_TYPE", std::string(inputTypeStr)));
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_VAL", "convert_float(score_threshold[0])"));
    } else {
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_VAL", "0.0f"));
    }

    if (params.has_soft_nms_sigma) {
        char inputTypeStr[128];
        snprintf(inputTypeStr, sizeof(inputTypeStr), "INPUT%d_TYPE", params.GetIndexSoftNmsSigma());
        jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_TYPE", std::string(inputTypeStr)));
        jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_VAL", "convert_float(soft_nms_sigma[0])"));
    } else {
        jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_VAL", "0.0f"));
    }

    if (params.has_second_output) {
        char inputTypeStr[128];
        snprintf(inputTypeStr, sizeof(inputTypeStr), "INPUT%d_TYPE", params.GetIndexSecondOutput());
        jit.AddConstant(MakeJitConstant("SECOND_OUTPUT_TYPE", std::string(inputTypeStr)));
        snprintf(inputTypeStr, sizeof(inputTypeStr), "TO_INPUT%d_TYPE", params.GetIndexSecondOutput());
        jit.AddConstant(MakeJitConstant("TO_SECOND_OUTPUT_TYPE", std::string(inputTypeStr)));
    }

    if (params.has_third_output) {
        char inputTypeStr[128];
        snprintf(inputTypeStr, sizeof(inputTypeStr), "INPUT%d_TYPE", params.GetIndexThirdOutput());
        jit.AddConstant(MakeJitConstant("THIRD_OUTPUT_TYPE", std::string(inputTypeStr)));
        snprintf(inputTypeStr, sizeof(inputTypeStr), "TO_INPUT%d_TYPE", params.GetIndexThirdOutput());
        jit.AddConstant(MakeJitConstant("TO_THIRD_OUTPUT_TYPE", std::string(inputTypeStr)));
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
    const size_t optimal_values[] = {16, 8, 7, 6, 5, 4, 2, 1};
    const size_t splitNum = gws[2];
    const size_t globalClassNum = gws[1];
    const auto rest_lws = info.maxWorkGroupSize / splitNum;
    size_t lws_idx = 0;
    while (rest_lws < optimal_values[lws_idx]) lws_idx++;
    while (globalClassNum % optimal_values[lws_idx]) lws_idx++;

    return optimal_values[lws_idx];
}

NonMaxSuppressionKernelRef::DispatchData SetDefault(const non_max_suppression_params& params, int idx) {
    NonMaxSuppressionKernelRef::DispatchData dispatchData;

    const auto& input = params.inputs[1];
    if (idx == 0) {
        dispatchData.gws = {input.Batch().v, input.Feature().v, 256};
        dispatchData.lws = {1, 1, 256};
    } else if (idx == 1) {
        const size_t kSplitNum = 16;
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
    if (idx == 0) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2 });

        if (params.has_score_threshold)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexScoreThreshold() });
    } else if (idx == 1) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2 });
    } else if (idx == 2) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2 });

        if (params.has_num_select_per_class)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexNumSelectPerClass() });
        if (params.has_iou_threshold)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexIouThreshold() });
        if (params.has_score_threshold)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexScoreThreshold() });
        if (params.has_soft_nms_sigma)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexSoftNmsSigma() });
    } else if (idx == 3) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });

        if (params.has_second_output)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexSecondOutput() });
        if (params.has_third_output)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexThirdOutput() });
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
    constexpr size_t intermidiate_bytes = 12;   // struct size of SortedBoxInfo/BoxInfo in non_max_suppression_gpu_ref.cl
    auto batch_num = orgParams.inputs[1].Batch().v;
    auto class_num = orgParams.inputs[1].Feature().v;
    auto boxes_num = orgParams.inputs[0].Feature().v;
    size_t buffer_stride = boxes_num * intermidiate_bytes;
    size_t buffer_size = batch_num * class_num * buffer_stride;
    size_t sel_num_buffer_size = batch_num * class_num * 4;

    kd.internalBufferSizes.push_back(buffer_size);
    kd.internalBufferSizes.push_back(buffer_size);
    kd.internalBufferSizes.push_back(sel_num_buffer_size);
    kd.internalBufferDataType = Datatype::F32;

    // Build clKernelData.
    for (size_t i = 0; i < kKernelsNum; i++) {
        DispatchData dispatchData = SetDefault(orgParams, static_cast<int>(i));
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
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
                                   , MakeJitConstant("IS_STAGE_0", "true")});
        } else if (i == 1) {
            cldnn_jit.AddConstants({ MakeJitConstant("IS_STAGE_1", "true")
                                   , MakeJitConstant("LOCAL_BATCH_NUM", dispatchData.lws[0])
                                   , MakeJitConstant("LOCAL_CLASS_NUM", dispatchData.lws[1])
                                   , MakeJitConstant("LOCAL_WORK_NUM", dispatchData.lws[2])
                                   , MakeJitConstant("PARTITION_STEP", GetPartitionStep(static_cast<int>(dispatchData.lws[2])))});
        } else if (i == 2) {
            cldnn_jit.AddConstant(MakeJitConstant("IS_STAGE_2", "true"));
        } else {
            cldnn_jit.AddConstant(MakeJitConstant("IS_STAGE_FINAL", "true"));
        }

        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[i];
        KernelBase::CheckDispatchData(kernelName, dispatchData);
        kernel.params.workGroups.global = dispatchData.gws;
        kernel.params.workGroups.local  = dispatchData.lws;
        kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
        SetKernelArguments(orgParams, kernel, i);
    }

    return {kd};
}

KernelsPriority NonMaxSuppressionKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
