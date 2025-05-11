// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matrix_nms_kernel_ref.h"

#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey MatrixNmsKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

namespace {

MatrixNmsKernelRef::DispatchData SetDefault(const matrix_nms_params& params, size_t idx) {
    MatrixNmsKernelRef::DispatchData dispatch_data;

    const auto& input_scores = params.inputs[1];
    if (idx == 0) {
        dispatch_data.gws = {input_scores.Batch().v, input_scores.Feature().v, 1};
        dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);
    } else if (idx == 1) {
        dispatch_data.gws = {input_scores.Batch().v, 1, 1};
        dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);
    } else {
        dispatch_data.gws = {1, 1, 1};
        dispatch_data.lws = {1, 1, 1};
    }

    return dispatch_data;
}

std::tuple<int, int> GetMaxBoxes(const matrix_nms_params& params) {
    const int classes_num = static_cast<const int>(params.inputs[1].Feature().v);
    const int boxes_num = static_cast<const int>(params.inputs[0].Feature().v);

    int max_boxes_per_class{boxes_num};
    if (params.nms_top_k >= 0)
        max_boxes_per_class = std::min(max_boxes_per_class, params.nms_top_k);

    auto classes_num_adj = classes_num;
    if (params.background_class >= 0 && params.background_class < classes_num)
        classes_num_adj = std::max(1, classes_num - 1);

    auto max_boxes_per_batch = max_boxes_per_class * classes_num_adj;
    if (params.keep_top_k >= 0)
        max_boxes_per_batch = std::min(max_boxes_per_batch, params.keep_top_k);

    return {max_boxes_per_class, max_boxes_per_batch};
}
}  // anonymous namespace

KernelsData MatrixNmsKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    constexpr size_t kernels_num{3};
    KernelData kernel_data = KernelData::Default<matrix_nms_params>(params, kernels_num);
    const matrix_nms_params& new_params = dynamic_cast<const matrix_nms_params&>(*kernel_data.params.get());

    constexpr size_t BOX_INFO_SIZE{16};

    const int batches_num = static_cast<const int>(new_params.inputs[1].Batch().v);
    const int classes_num = static_cast<const int>(new_params.inputs[1].Feature().v);

    int max_boxes_per_class, max_boxes_per_batch;
    std::tie(max_boxes_per_class, max_boxes_per_batch) = GetMaxBoxes(new_params);

    const size_t box_info_num = batches_num * classes_num * max_boxes_per_class;

    const size_t box_info_buffer_size = box_info_num * BOX_INFO_SIZE;
    const size_t sel_boxes_num_buffer_size = batches_num * classes_num * sizeof(int);

    kernel_data.internalBuffers.push_back(box_info_buffer_size);
    kernel_data.internalBuffers.push_back(sel_boxes_num_buffer_size);
    kernel_data.internalBufferDataType = Datatype::F32;

    for (size_t i{}; i < kernels_num; ++i) {
        auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, i);
        auto jit_constants = GetJitConstants(new_params);
        jit_constants.AddConstant(MakeJitConstant("MATRIX_NMS_STAGE_" + std::to_string(i), "true"));

        jit_constants.AddConstant(MakeJitConstant("MAX_BOXES_PER_CLASS", max_boxes_per_class));
        jit_constants.AddConstant(MakeJitConstant("MAX_BOXES_PER_BATCH", max_boxes_per_batch));
        auto jit = CreateJit(kernelName, jit_constants, entry_point);

        DispatchData dispatch_data = SetDefault(new_params, i);
        auto& kernel = kernel_data.kernels[i];
        KernelBase::CheckDispatchData(kernelName, dispatch_data, params.engineInfo.maxWorkGroupSize);
        kernel.params.workGroups.global = dispatch_data.gws;
        kernel.params.workGroups.local = dispatch_data.lws;
        kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);

        SetKernelArguments(new_params, kernel, i);
    }

    return {kernel_data};
}

float MatrixNmsKernelRef::GetKernelsPriority(const Params& params) const {
    return FORCE_PRIORITY_9;
}

bool MatrixNmsKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::MATRIX_NMS) {
        return false;
    }

    return true;
}

JitConstants MatrixNmsKernelRef::GetJitConstants(const matrix_nms_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& boxes = params.inputs[0];
    switch (boxes.GetDType()) {
    case Datatype::F32:
        jit.AddConstant(MakeJitConstant("COORD_TYPE_4", "float4"));
        jit.AddConstant(MakeJitConstant("TINY", "1e-10f"));
        break;

    case Datatype::F16:
        jit.AddConstant(MakeJitConstant("COORD_TYPE_4", "half4"));
        jit.AddConstant(MakeJitConstant("TINY", "1e-7h"));
        break;
        break;

    default:
        throw std::invalid_argument("Matrix NMS boxes type should be one of F32 or F16.");
    }

    jit.AddConstant(MakeJitConstant("SORT_TYPE", params.sort_type));
    jit.AddConstant(MakeJitConstant("SORT_RESULT_ACROSS_BATCH", params.sort_result_across_batch));
    jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD", params.score_threshold));
    jit.AddConstant(MakeJitConstant("KEEP_TOP_K", params.keep_top_k));
    jit.AddConstant(MakeJitConstant("BACKGROUND_CLASS", params.background_class));
    jit.AddConstant(MakeJitConstant("DECAY_FUNC", params.decay));
    jit.AddConstant(MakeJitConstant("GAUSSIAN_SIGMA", params.gaussian_sigma));
    jit.AddConstant(MakeJitConstant("POST_THRESHOLD", params.post_threshold));
    jit.AddConstant(MakeJitConstant("NORM", params.normalized ? "INPUT0_VAL_ZERO" : "INPUT0_VAL_ONE"));
    return jit;
}

void MatrixNmsKernelRef::SetKernelArguments(const matrix_nms_params& params, clKernelData& kernel, size_t idx) const {
    switch (idx) {
    case 0:
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        break;

    case 1:
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 2});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        break;

    case 2:
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 2});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        break;

    default:
        throw std::invalid_argument("Matrix NMS has 3 kernels. valid index is 0 ~ 2.");
    }
}

}  // namespace kernel_selector
