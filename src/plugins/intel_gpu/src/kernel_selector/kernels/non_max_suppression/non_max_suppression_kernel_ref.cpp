// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_max_suppression_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
namespace {
static inline int GetPartitionStep(int localWorkItemNum) {
    int step_size = 0;
    for (int temp = localWorkItemNum; temp > 1; temp /= 2) {
        step_size++;
    }
    return step_size;
}

static inline size_t GetOptimalLocalClassSize(std::vector<size_t> gws, const EngineInfo& info) {
    const size_t optimal_values[] = {256, 227, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 2, 1};
    const size_t splitNum = gws[2];
    const size_t globalClassNum = gws[1];
    const auto rest_lws = info.maxWorkGroupSize / splitNum;
    size_t lws_idx = 0;
    while (rest_lws < optimal_values[lws_idx])
        lws_idx++;
    while (globalClassNum % optimal_values[lws_idx])
        lws_idx++;

    return optimal_values[lws_idx];
}

NonMaxSuppressionKernelRef::DispatchData SetDefault(const non_max_suppression_params& params, int idx) {
    NonMaxSuppressionKernelRef::DispatchData dispatchData;

    const auto& input = params.inputs[1];
    if (idx == 0) {
        const size_t boxesGroupSize = std::min(params.inputs[0].Feature().v, params.engineInfo.maxWorkGroupSize);
        dispatchData.gws = {input.Batch().v, input.Feature().v, boxesGroupSize};
        dispatchData.lws = {1, 1, boxesGroupSize};
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
}  // namespace
Datatype NonMaxSuppressionKernelRef::GetAccumulatorType(const non_max_suppression_params& params) const {
    auto in_dt = params.inputs[0].GetDType();
    auto out_dt = params.outputs[0].GetDType();

    auto smaller_fp_type = [](const Datatype& current, const Datatype& candidate) -> Datatype {
        if (candidate != Datatype::F32 && candidate != Datatype::F16)
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

inline std::string GetInputTypeStr(uint32_t idx) {
    return "INPUT" + std::to_string(idx) + "_TYPE";
}

inline std::string GetToInputTypeStr(uint32_t idx) {
    return "TO_" + GetInputTypeStr(idx);
}

inline std::string GetToInputIndexStr(uint32_t idx) {
    return "INPUT" + std::to_string(idx) + "_GET_INDEX";
}

JitConstants NonMaxSuppressionKernelRef::GetJitConstants(const non_max_suppression_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& input0 = params.inputs[0];
    switch (input0.GetDType()) {
    case Datatype::F32:
        jit.AddConstant(MakeJitConstant("TO_COORD_TYPE_4", "convert_float4"));
        jit.AddConstant(MakeJitConstant("COORD_TYPE_4", "float4"));
        jit.AddConstant(MakeJitConstant("TO_COORD_TYPE", "convert_float"));
        jit.AddConstant(MakeJitConstant("COORD_TYPE", "float"));
        break;

    case Datatype::F16:
        jit.AddConstant(MakeJitConstant("TO_COORD_TYPE_4", "convert_half4"));
        jit.AddConstant(MakeJitConstant("COORD_TYPE_4", "half4"));
        jit.AddConstant(MakeJitConstant("TO_COORD_TYPE", "convert_half"));
        jit.AddConstant(MakeJitConstant("COORD_TYPE", "half"));
        break;

    default:
        throw std::invalid_argument("NMS input0 type should be one of F32 or F16.");
    }

    jit.AddConstants({MakeJitConstant("SORT_RESULT_DESCENDING", params.sort_result_descending),
                      MakeJitConstant("BOX_ENCODING", static_cast<int>(params.box_encoding))});

    jit.AddConstant(MakeJitConstant("OUTPUT_NUM", params.outputs[0].Batch().v));

    if (params.num_select_per_class_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_TYPE", GetInputTypeStr(params.GetIndexNumSelectPerClass())));
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_VAL", "convert_int(num_select_per_class[0])"));
    } else {
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_VAL", params.num_select_per_class));
    }

    if (params.iou_threshold_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_TYPE", GetInputTypeStr(params.GetIndexIouThreshold())));
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_VAL", "convert_float(iou_threshold[0])"));
    } else {
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_VAL", params.iou_threshold));
    }

    if (params.score_threshold_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_TYPE", GetInputTypeStr(params.GetIndexScoreThreshold())));
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_VAL", "convert_float(score_threshold[0])"));
    } else {
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_VAL", params.score_threshold));
    }

    if (params.rotation == NMSRotationType::NONE) {
        if (params.soft_nms_sigma_type == base_params::ArgType::Input) {
            jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_TYPE", GetInputTypeStr(params.GetIndexSoftNmsSigma())));
            jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_VAL", "convert_float(soft_nms_sigma[0])"));
        } else {
            jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_VAL", params.soft_nms_sigma));
        }
    } else {
        jit.AddConstant(MakeJitConstant("ROTATION", static_cast<int>(params.rotation)));
        // for NMSRotated it is always zero
        jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_VAL", 0.0f));
    }

    return jit;
}

bool NonMaxSuppressionKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::NON_MAX_SUPPRESSION) {
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
 *  INTERNAL_BUFFER[2]: intermidiate_sorted_box_num
 *
 */
void NonMaxSuppressionKernelRef::SetKernelArguments(const non_max_suppression_params& params,
                                                    clKernelData& kernel, size_t idx) const {
    switch (idx) {
    case 0:
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2 });
        if (params.score_threshold_type == base_params::ArgType::Input)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexScoreThreshold() });
        break;

    case 1:
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2 });
        break;

    case 2:
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 2 });

        if (params.num_select_per_class_type == base_params::ArgType::Input)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexNumSelectPerClass() });
        if (params.iou_threshold_type == base_params::ArgType::Input)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexIouThreshold() });
        if (params.score_threshold_type == base_params::ArgType::Input)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexScoreThreshold() });
        if (params.soft_nms_sigma_type == base_params::ArgType::Input)
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, params.GetIndexSoftNmsSigma() });
        break;

    case 3:
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 1 });
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, 0 });
        for (size_t i = 1; i < params.outputs.size(); i++) {
            kernel.params.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, static_cast<uint32_t>(i) });
        }

        break;

    default:
        throw std::invalid_argument("NMS has 4 kernels. valid index is 0 ~ 3.");
    }
}

KernelsData NonMaxSuppressionKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    constexpr size_t kKernelsNum = 4;
    KernelData kd = KernelData::Default<non_max_suppression_params>(params, kKernelsNum);
    const non_max_suppression_params& orgParams = static_cast<const non_max_suppression_params&>(params);

    // Assign internel buffer
    constexpr size_t intermediate_bytes = 12;   // struct size of SortedBoxInfo/BoxInfo in non_max_suppression_gpu_ref.cl
    auto batch_num = orgParams.inputs[1].Batch().v;
    auto class_num = orgParams.inputs[1].Feature().v;
    auto boxes_num = orgParams.inputs[0].Feature().v;
    size_t buffer_stride = boxes_num * intermediate_bytes;
    size_t buffer_size = batch_num * class_num * buffer_stride;
    size_t sel_num_buffer_size = batch_num * class_num * sizeof(int);

    kd.internalBuffers.push_back(buffer_size);
    kd.internalBuffers.push_back(buffer_size);
    kd.internalBuffers.push_back(sel_num_buffer_size);
    kd.internalBufferDataType = Datatype::F32;

    // Build clKernelData.
    for (size_t i = 0; i < kKernelsNum; i++) {
        DispatchData dispatchData = SetDefault(orgParams, static_cast<int>(i));
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, i);
        auto cldnn_jit = GetJitConstants(orgParams);
        cldnn_jit.AddConstant(MakeJitConstant("BUFFER_STRIDE", buffer_stride));

        if (i == 0) {
            size_t num_bit_mask = CeilDiv(boxes_num, 8);
            size_t num_score_per_item = RoundUp(CeilDiv(boxes_num, params.engineInfo.maxWorkGroupSize), 8);
            size_t num_score_block = CeilDiv(boxes_num, num_score_per_item);
            cldnn_jit.AddConstants({ MakeJitConstant("NUM_BIT_MASK", num_bit_mask)
                                   , MakeJitConstant("NUM_SCORE_PER_ITEM", num_score_per_item)
                                   , MakeJitConstant("NUM_SCORE_BLOCK", num_score_block)});
        } else if (i == 1) {
            cldnn_jit.AddConstants({ MakeJitConstant("LOCAL_BATCH_NUM", dispatchData.lws[0])
                                   , MakeJitConstant("LOCAL_CLASS_NUM", dispatchData.lws[1])
                                   , MakeJitConstant("LOCAL_WORK_NUM", dispatchData.lws[2])
                                   , MakeJitConstant("PARTITION_STEP", GetPartitionStep(static_cast<int>(dispatchData.lws[2])))});
        } else if (i == 2 && orgParams.reuse_internal_buffer) {
            cldnn_jit.AddConstant({ MakeJitConstant("REUSE_INTERNAL_BUFFER", 1)});
        }
        cldnn_jit.AddConstant(MakeJitConstant("NMS_STAGE_" + std::to_string(i), "true"));

        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[i];
        KernelBase::CheckDispatchData(kernelName, dispatchData, params.engineInfo.maxWorkGroupSize);
        kernel.params.workGroups.global = dispatchData.gws;
        kernel.params.workGroups.local  = dispatchData.lws;
        kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
        SetKernelArguments(orgParams, kernel, i);
    }

    return {kd};
}

KernelsPriority NonMaxSuppressionKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
