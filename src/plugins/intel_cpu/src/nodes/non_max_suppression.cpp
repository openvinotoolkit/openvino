// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (c) Facebook, Inc. and its affiliates.
// The implementation for rotated boxes intersection is based on the code from:
// https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h
//

#include "non_max_suppression.h"

#include <queue>

#include "cpu_types.h"
#include "openvino/core/parallel.hpp"
#include "openvino/op/nms_rotated.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool NonMaxSuppression::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                             std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    op::v9::NonMaxSuppression::get_type_info_static(),
                    op::internal::NonMaxSuppressionIEInternal::get_type_info_static(),
                    op::v13::NMSRotated::get_type_info_static())) {
            errorMessage = "Only NonMaxSuppression from opset9, NonMaxSuppressionIEInternal and NMSRotated from "
                           "opset13 are supported.";
            return false;
        }

        if (auto nms9 = as_type<const op::v9::NonMaxSuppression>(op.get())) {
            const auto boxEncoding = nms9->get_box_encoding();
            if (!one_of(boxEncoding,
                        op::v9::NonMaxSuppression::BoxEncodingType::CENTER,
                        op::v9::NonMaxSuppression::BoxEncodingType::CORNER)) {
                errorMessage = "Supports only CENTER and CORNER box encoding type";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

NonMaxSuppression::NonMaxSuppression(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (one_of(op->get_type_info(), op::internal::NonMaxSuppressionIEInternal::get_type_info_static())) {
        m_out_static_shape = true;
    }

    if (getOriginalInputsNumber() < 2 || getOriginalInputsNumber() > NMS_SOFT_NMS_SIGMA + 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges: ", getOriginalInputsNumber());
    }
    if (getOriginalOutputsNumber() != 3) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges: ", getOriginalOutputsNumber());
    }

    if (auto nms9 = as_type<const op::v9::NonMaxSuppression>(op.get())) {
        boxEncodingType = static_cast<NMSBoxEncodeType>(nms9->get_box_encoding());
        m_sort_result_descending = nms9->get_sort_result_descending();
        m_coord_num = 4lu;
    } else if (auto nmsIe = as_type<const op::internal::NonMaxSuppressionIEInternal>(op.get())) {
        boxEncodingType = nmsIe->m_center_point_box ? NMSBoxEncodeType::CENTER : NMSBoxEncodeType::CORNER;
        m_sort_result_descending = nmsIe->m_sort_result_descending;
        m_coord_num = 4lu;
    } else if (auto nms = as_type<const op::v13::NMSRotated>(op.get())) {
        m_sort_result_descending = nms->get_sort_result_descending();
        m_clockwise = nms->get_clockwise();
        m_rotated_boxes = true;
        m_coord_num = 5lu;
    } else {
        const auto& typeInfo = op->get_type_info();
        THROW_CPU_NODE_ERR("doesn't support NMS: ", typeInfo.name, " v", typeInfo.version_id);
    }

    const auto& boxes_dims = getInputShapeAtPort(NMS_BOXES).getDims();
    if (boxes_dims.size() != 3) {
        THROW_CPU_NODE_ERR("has unsupported 'boxes' input rank: ", boxes_dims.size());
    }
    if (boxes_dims[2] != m_coord_num) {
        THROW_CPU_NODE_ERR("has unsupported 'boxes' input 3rd dimension size: ", boxes_dims[2]);
    }

    const auto& scores_dims = getInputShapeAtPort(NMS_SCORES).getDims();
    if (scores_dims.size() != 3) {
        THROW_CPU_NODE_ERR("has unsupported 'scores' input rank: ", scores_dims.size());
    }

    const auto& valid_outputs_shape = getOutputShapeAtPort(NMS_VALID_OUTPUTS);
    if (valid_outputs_shape.getRank() != 1) {
        THROW_CPU_NODE_ERR("has unsupported 'valid_outputs' output rank: ", valid_outputs_shape.getRank());
    }
    if (valid_outputs_shape.getDims()[0] != 1) {
        THROW_CPU_NODE_ERR("has unsupported 'valid_outputs' output 1st dimension size: ",
                           valid_outputs_shape.getDims()[1]);
    }

    for (size_t i = 0lu; i < op->get_output_size(); i++) {
        m_defined_outputs[i] = !op->get_output_target_inputs(i).empty();
    }
}

void NonMaxSuppression::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    const auto inputs_num = inputShapes.size();
    if (inputs_num > NMS_MAX_OUTPUT_BOXES_PER_CLASS) {
        check1DInput(getInputShapeAtPort(NMS_MAX_OUTPUT_BOXES_PER_CLASS),
                     "max_output_boxes_per_class",
                     NMS_MAX_OUTPUT_BOXES_PER_CLASS);
    }
    if (inputs_num > NMS_IOU_THRESHOLD) {
        check1DInput(getInputShapeAtPort(NMS_IOU_THRESHOLD), "iou_threshold", NMS_IOU_THRESHOLD);
    }
    if (inputs_num > NMS_SCORE_THRESHOLD) {
        check1DInput(getInputShapeAtPort(NMS_SCORE_THRESHOLD), "score_threshold", NMS_SCORE_THRESHOLD);
    }
    if (inputs_num > NMS_SOFT_NMS_SIGMA) {
        check1DInput(getInputShapeAtPort(NMS_SCORE_THRESHOLD), "soft_nms_sigma", NMS_SCORE_THRESHOLD);
    }

    checkOutput(getOutputShapeAtPort(NMS_SELECTED_INDICES), "selected_indices", NMS_SELECTED_INDICES);
    checkOutput(getOutputShapeAtPort(NMS_SELECTED_SCORES), "selected_scores", NMS_SELECTED_SCORES);

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputs_num);
    for (size_t i = 0; i < inputs_num; ++i) {
        ov::element::Type inPrecision = i == NMS_MAX_OUTPUT_BOXES_PER_CLASS ? ov::element::i32 : ov::element::f32;
        inDataConf.emplace_back(LayoutType::ncsp, inPrecision);
    }

    std::vector<PortConfigurator> outDataConf;
    outDataConf.reserve(outputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); ++i) {
        ov::element::Type outPrecision = i == NMS_SELECTED_SCORES ? ov::element::f32 : ov::element::i32;
        outDataConf.emplace_back(LayoutType::ncsp, outPrecision);
    }

    impl_desc_type impl_type = impl_desc_type::ref;

#if defined(OPENVINO_ARCH_X86_64)
    using namespace dnnl::impl::cpu;

    // As only FP32 and ncsp is supported, and kernel is shape agnostic, we can create here. There is no need to
    // recompilation.
    createJitKernel();

    x64::cpu_isa_t actual_isa = x64::isa_undef;
    if (m_jit_kernel) {
        actual_isa = m_jit_kernel->getIsa();
    }
    switch (actual_isa) {
    case x64::avx512_core:
        impl_type = impl_desc_type::jit_avx512;
        break;
    case x64::avx2:
        impl_type = impl_desc_type::jit_avx2;
        break;
    case x64::sse41:
        impl_type = impl_desc_type::jit_sse42;
        break;
    default:
        impl_type = impl_desc_type::ref;
    }
#endif  // OPENVINO_ARCH_X86_64

    addSupportedPrimDesc(inDataConf, outDataConf, impl_type);
}

void NonMaxSuppression::prepareParams() {
    const auto& boxesDims = isDynamicNode() ? getParentEdgeAt(NMS_BOXES)->getMemory().getStaticDims()
                                            : getInputShapeAtPort(NMS_BOXES).getStaticDims();
    const auto& scoresDims = isDynamicNode() ? getParentEdgeAt(NMS_SCORES)->getMemory().getStaticDims()
                                             : getInputShapeAtPort(NMS_SCORES).getStaticDims();

    m_batches_num = boxesDims[0];
    m_boxes_num = boxesDims[1];
    m_classes_num = scoresDims[1];
    if (m_batches_num != scoresDims[0]) {
        THROW_CPU_NODE_ERR("Batches number is different in 'boxes' and 'scores' inputs");
    }
    if (m_boxes_num != scoresDims[2]) {
        THROW_CPU_NODE_ERR("Boxes number is different in 'boxes' and 'scores' inputs");
    }

    m_output_boxes_per_class = std::min(m_max_output_boxes_per_class, m_boxes_num);
    const auto max_number_of_boxes = m_output_boxes_per_class * m_batches_num * m_classes_num;
    m_filtered_boxes.resize(max_number_of_boxes);

    m_num_filtered_boxes.resize(m_batches_num);
    for (auto& i : m_num_filtered_boxes) {
        i.resize(m_classes_num);
    }
}

void NonMaxSuppression::createJitKernel() {
#if defined(OPENVINO_ARCH_X86_64)
    if (!m_rotated_boxes) {
        auto jcp = kernel::NmsCompileParams();
        jcp.box_encode_type = boxEncodingType;
        jcp.is_soft_suppressed_by_iou = m_is_soft_suppressed_by_iou;

        m_jit_kernel =
            kernel::JitKernel<kernel::NmsCompileParams, kernel::NmsCallArgs>::createInstance<kernel::NonMaxSuppression>(
                jcp);
    }
#endif  // OPENVINO_ARCH_X86_64
}

void NonMaxSuppression::executeDynamicImpl(const dnnl::stream& strm) {
    if (hasEmptyInputTensors() || (inputShapes.size() > NMS_MAX_OUTPUT_BOXES_PER_CLASS &&
                                   getSrcDataAtPortAs<int>(NMS_MAX_OUTPUT_BOXES_PER_CLASS)[0] == 0)) {
        redefineOutputMemory({{0, 3}, {0, 3}, {1}});
        *getDstDataAtPortAs<int>(NMS_VALID_OUTPUTS) = 0;
        return;
    }
    execute(strm);
}

void NonMaxSuppression::execute(const dnnl::stream& strm) {
    const auto inputs_num = inputShapes.size();

    size_t max_number_of_boxes = m_output_boxes_per_class * m_batches_num * m_classes_num;
    if (inputs_num > NMS_MAX_OUTPUT_BOXES_PER_CLASS) {
        auto val = getSrcDataAtPortAs<int32_t>(NMS_MAX_OUTPUT_BOXES_PER_CLASS)[0];
        m_max_output_boxes_per_class = val <= 0l ? 0lu : static_cast<size_t>(val);
        m_output_boxes_per_class = std::min(m_max_output_boxes_per_class, m_boxes_num);
        max_number_of_boxes = m_output_boxes_per_class * m_batches_num * m_classes_num;
        m_filtered_boxes.resize(max_number_of_boxes);
    }
    if (m_max_output_boxes_per_class == 0lu) {
        return;
    }

    if (inputs_num > NMS_IOU_THRESHOLD) {
        m_iou_threshold = getSrcDataAtPortAs<float>(NMS_IOU_THRESHOLD)[0];
    }
    if (inputs_num > NMS_SCORE_THRESHOLD) {
        m_score_threshold = getSrcDataAtPortAs<float>(NMS_SCORE_THRESHOLD)[0];
    }
    if (inputs_num > NMS_SOFT_NMS_SIGMA) {
        m_soft_nms_sigma = getSrcDataAtPortAs<float>(NMS_SOFT_NMS_SIGMA)[0];
        m_scale = (m_soft_nms_sigma > 0.f) ? (-0.5f / m_soft_nms_sigma) : 0.f;
    }

    auto boxes_memory = getSrcMemoryAtPort(NMS_BOXES);
    auto scores_memory = getSrcMemoryAtPort(NMS_SCORES);

    auto boxes = boxes_memory->getDataAs<const float>();
    auto scores = scores_memory->getDataAs<const float>();

    const auto& boxes_strides = boxes_memory->getDescWithType<BlockedMemoryDesc>()->getStrides();
    const auto& scores_strides = scores_memory->getDescWithType<BlockedMemoryDesc>()->getStrides();

    if (m_rotated_boxes) {
        nmsRotated(boxes, scores, boxes_strides, scores_strides, m_filtered_boxes);
    } else if (m_soft_nms_sigma == 0.f) {
        nmsWithoutSoftSigma(boxes, scores, boxes_strides, scores_strides, m_filtered_boxes);
    } else {
        nmsWithSoftSigma(boxes, scores, boxes_strides, scores_strides, m_filtered_boxes);
    }

    size_t start_offset = m_num_filtered_boxes[0][0];
    for (size_t b = 0lu; b < m_num_filtered_boxes.size(); b++) {
        size_t batchOffset = b * m_classes_num * m_output_boxes_per_class;
        for (size_t c = (b == 0lu ? 1lu : 0lu); c < m_num_filtered_boxes[b].size(); c++) {
            size_t offset = batchOffset + c * m_output_boxes_per_class;
            for (size_t i = 0lu; i < m_num_filtered_boxes[b][c]; i++) {
                m_filtered_boxes[start_offset + i] = m_filtered_boxes[offset + i];
            }
            start_offset += m_num_filtered_boxes[b][c];
        }
    }

    auto boxes_ptr = m_filtered_boxes.data();
    // need more particular comparator to get deterministic behaviour
    // escape situation when filtred boxes with same score have different position from launch to launch
    if (m_sort_result_descending) {
        parallel_sort(boxes_ptr, boxes_ptr + start_offset, [](const FilteredBox& l, const FilteredBox& r) {
            return (l.score > r.score) || (l.score == r.score && l.batch_index < r.batch_index) ||
                   (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                   (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index &&
                    l.box_index < r.box_index);
        });
    }

    const size_t valid_outputs = std::min(start_offset, max_number_of_boxes);

    const size_t stride = 3lu;
    if (!m_out_static_shape) {
        VectorDims new_dims{valid_outputs, stride};
        redefineOutputMemory({new_dims, new_dims, {1}});
    }

    if (m_defined_outputs[NMS_SELECTED_INDICES]) {
        auto out_ptr = getDstDataAtPortAs<int32_t>(NMS_SELECTED_INDICES);
        int32_t* boxes_ptr = &(m_filtered_boxes[0].batch_index);

        size_t idx = 0lu;
        for (; idx < valid_outputs; idx++) {
            memcpy(out_ptr, boxes_ptr, 12);
            out_ptr += stride;
            boxes_ptr += 4;
        }

        if (m_out_static_shape) {
            std::fill(out_ptr, out_ptr + (max_number_of_boxes - idx) * stride, -1);
        }
    }

    if (m_defined_outputs[NMS_SELECTED_SCORES]) {
        auto out_ptr = getDstDataAtPortAs<float>(NMS_SELECTED_SCORES);

        size_t idx = 0lu;
        for (; idx < valid_outputs; idx++) {
            out_ptr[0] = static_cast<float>(m_filtered_boxes[idx].batch_index);
            out_ptr[1] = static_cast<float>(m_filtered_boxes[idx].class_index);
            out_ptr[2] = m_filtered_boxes[idx].score;
            out_ptr += stride;
        }

        if (m_out_static_shape) {
            std::fill(out_ptr, out_ptr + (max_number_of_boxes - idx) * stride, -1.f);
        }
    }

    if (m_defined_outputs[NMS_VALID_OUTPUTS]) {
        auto out_ptr = getDstDataAtPortAs<int32_t>(NMS_VALID_OUTPUTS);
        *out_ptr = static_cast<int32_t>(valid_outputs);
    }
}

void NonMaxSuppression::nmsWithSoftSigma(const float* boxes,
                                         const float* scores,
                                         const VectorDims& boxesStrides,
                                         const VectorDims& scoresStrides,
                                         std::vector<FilteredBox>& filtBoxes) {
    auto less = [](const boxInfo& l, const boxInfo& r) {
        return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
    };

    // update score, if iou is 0, weight is 1, score does not change
    // if is_soft_suppressed_by_iou is false, apply for all iou, including iou>iou_threshold, soft suppressed when score
    // < score_threshold if is_soft_suppressed_by_iou is true, hard suppressed by iou_threshold, then soft suppress
    auto coeff = [&](float iou) {
        if (m_is_soft_suppressed_by_iou && iou > m_iou_threshold) {
            return 0.0f;
        }
        return std::exp(m_scale * iou * iou);
    };

    parallel_for2d(m_batches_num, m_classes_num, [&](int batch_idx, int class_idx) {
        std::vector<FilteredBox> selectedBoxes;
        const float* boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float* scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(
            less);  // score, box_id, suppress_begin_index
        for (int box_idx = 0; box_idx < static_cast<int>(m_boxes_num); box_idx++) {
            if (scoresPtr[box_idx] > m_score_threshold) {
                sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
            }
        }
        size_t sorted_boxes_size = sorted_boxes.size();
        size_t maxSeletedBoxNum = std::min(sorted_boxes_size, m_output_boxes_per_class);
        selectedBoxes.reserve(maxSeletedBoxNum);
        if (maxSeletedBoxNum > 0) {
            // include first directly
            boxInfo candidateBox = sorted_boxes.top();
            sorted_boxes.pop();
            selectedBoxes.emplace_back(candidateBox.score, batch_idx, class_idx, candidateBox.idx);
            if (maxSeletedBoxNum > 1) {
                if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
                    std::vector<float> boxCoord0(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord1(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord2(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord3(maxSeletedBoxNum, 0.0f);

                    boxCoord0[0] = boxesPtr[candidateBox.idx * m_coord_num];
                    boxCoord1[0] = boxesPtr[candidateBox.idx * m_coord_num + 1];
                    boxCoord2[0] = boxesPtr[candidateBox.idx * m_coord_num + 2];
                    boxCoord3[0] = boxesPtr[candidateBox.idx * m_coord_num + 3];

                    auto arg = kernel::NmsCallArgs();
                    arg.iou_threshold = static_cast<float*>(&m_iou_threshold);
                    arg.score_threshold = static_cast<float*>(&m_score_threshold);
                    arg.scale = static_cast<float*>(&m_scale);
                    while (selectedBoxes.size() < m_output_boxes_per_class && !sorted_boxes.empty()) {
                        boxInfo candidateBox = sorted_boxes.top();
                        float origScore = candidateBox.score;
                        sorted_boxes.pop();

                        int candidateStatus =
                            NMSCandidateStatus::SELECTED;  // 0 for suppressed, 1 for selected, 2 for updated
                        arg.score = static_cast<float*>(&candidateBox.score);
                        arg.selected_boxes_num = selectedBoxes.size() - candidateBox.suppress_begin_index;
                        arg.selected_boxes_coord[0] =
                            static_cast<float*>(&boxCoord0[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[1] =
                            static_cast<float*>(&boxCoord1[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[2] =
                            static_cast<float*>(&boxCoord2[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[3] =
                            static_cast<float*>(&boxCoord3[candidateBox.suppress_begin_index]);
                        arg.candidate_box = static_cast<const float*>(&boxesPtr[candidateBox.idx * m_coord_num]);
                        arg.candidate_status = static_cast<int*>(&candidateStatus);
                        (*m_jit_kernel)(&arg);

                        if (candidateStatus == NMSCandidateStatus::SUPPRESSED) {
                            continue;
                        }
                        if (candidateBox.score == origScore) {
                            selectedBoxes.emplace_back(candidateBox.score, batch_idx, class_idx, candidateBox.idx);
                            int selectedSize = selectedBoxes.size();
                            boxCoord0[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num];
                            boxCoord1[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 1];
                            boxCoord2[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 2];
                            boxCoord3[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 3];
                        } else {
                            if (candidateBox.score == origScore) {
                                selectedBoxes.emplace_back(candidateBox.score, batch_idx, class_idx, candidateBox.idx);
                                int selectedSize = selectedBoxes.size();
                                boxCoord0[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num];
                                boxCoord1[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 1];
                                boxCoord2[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 2];
                                boxCoord3[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 3];
                            } else {
                                candidateBox.suppress_begin_index = selectedBoxes.size();
                                sorted_boxes.push(candidateBox);
                            }
                        }
                    }
#endif  // OPENVINO_ARCH_X86_64
                } else {
                    while (selectedBoxes.size() < m_output_boxes_per_class && !sorted_boxes.empty()) {
                        boxInfo candidateBox = sorted_boxes.top();
                        float origScore = candidateBox.score;
                        sorted_boxes.pop();

                        int candidateStatus =
                            NMSCandidateStatus::SELECTED;  // 0 for suppressed, 1 for selected, 2 for updated
                        for (int selected_idx = static_cast<int>(selectedBoxes.size()) - 1;
                             selected_idx >= candidateBox.suppress_begin_index;
                             selected_idx--) {
                            float iou =
                                intersectionOverUnion(&boxesPtr[candidateBox.idx * m_coord_num],
                                                      &boxesPtr[selectedBoxes[selected_idx].box_index * m_coord_num]);

                            // when is_soft_suppressed_by_iou is true, score is decayed to zero and implicitely
                            // suppressed if iou > iou_threshold.
                            candidateBox.score *= coeff(iou);
                            // soft suppressed
                            if (candidateBox.score <= m_score_threshold) {
                                candidateStatus = NMSCandidateStatus::SUPPRESSED;
                                break;
                            }
                        }

                        if (candidateStatus == NMSCandidateStatus::SUPPRESSED) {
                            continue;
                        }
                        if (candidateBox.score == origScore) {
                            selectedBoxes.emplace_back(candidateBox.score, batch_idx, class_idx, candidateBox.idx);
                        } else {
                            if (candidateBox.score == origScore) {
                                selectedBoxes.emplace_back(candidateBox.score, batch_idx, class_idx, candidateBox.idx);
                            } else {
                                candidateBox.suppress_begin_index = selectedBoxes.size();
                                sorted_boxes.push(candidateBox);
                            }
                        }
                    }
                }
            }
        }
        m_num_filtered_boxes[batch_idx][class_idx] = selectedBoxes.size();
        size_t offset = batch_idx * m_classes_num * m_output_boxes_per_class + class_idx * m_output_boxes_per_class;
        for (size_t i = 0; i < selectedBoxes.size(); i++) {
            filtBoxes[offset + i] = selectedBoxes[i];
        }
    });
}

void NonMaxSuppression::nmsWithoutSoftSigma(const float* boxes,
                                            const float* scores,
                                            const VectorDims& boxesStrides,
                                            const VectorDims& scoresStrides,
                                            std::vector<FilteredBox>& filtBoxes) {
    auto max_out_box = static_cast<int>(m_output_boxes_per_class);
    parallel_for2d(m_batches_num, m_classes_num, [&](int batch_idx, int class_idx) {
        const float* boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float* scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::vector<std::pair<float, int>> sorted_boxes;  // score, box_idx
        sorted_boxes.reserve(m_boxes_num);
        for (size_t box_idx = 0; box_idx < m_boxes_num; box_idx++) {
            if (scoresPtr[box_idx] > m_score_threshold) {
                sorted_boxes.emplace_back(scoresPtr[box_idx], box_idx);
            }
        }

        int io_selection_size = 0;
        const size_t sortedBoxSize = sorted_boxes.size();
        if (sortedBoxSize > 0lu) {
            parallel_sort(sorted_boxes.begin(),
                          sorted_boxes.end(),
                          [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                              return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                          });
            int offset = batch_idx * m_classes_num * m_output_boxes_per_class + class_idx * m_output_boxes_per_class;
            filtBoxes[offset + 0] = FilteredBox(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
            io_selection_size++;
            if (sortedBoxSize > 1lu) {
                if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
                    std::vector<float> boxCoord0(sortedBoxSize, 0.0f);
                    std::vector<float> boxCoord1(sortedBoxSize, 0.0f);
                    std::vector<float> boxCoord2(sortedBoxSize, 0.0f);
                    std::vector<float> boxCoord3(sortedBoxSize, 0.0f);

                    boxCoord0[0] = boxesPtr[sorted_boxes[0].second * m_coord_num];
                    boxCoord1[0] = boxesPtr[sorted_boxes[0].second * m_coord_num + 1];
                    boxCoord2[0] = boxesPtr[sorted_boxes[0].second * m_coord_num + 2];
                    boxCoord3[0] = boxesPtr[sorted_boxes[0].second * m_coord_num + 3];

                    auto arg = kernel::NmsCallArgs();
                    arg.iou_threshold = static_cast<float*>(&m_iou_threshold);
                    arg.score_threshold = static_cast<float*>(&m_score_threshold);
                    arg.scale = static_cast<float*>(&m_scale);
                    // box start index do not change for hard supresion
                    arg.selected_boxes_coord[0] = static_cast<float*>(&boxCoord0[0]);
                    arg.selected_boxes_coord[1] = static_cast<float*>(&boxCoord1[0]);
                    arg.selected_boxes_coord[2] = static_cast<float*>(&boxCoord2[0]);
                    arg.selected_boxes_coord[3] = static_cast<float*>(&boxCoord3[0]);

                    for (size_t candidate_idx = 1; (candidate_idx < sortedBoxSize) && (io_selection_size < max_out_box);
                         candidate_idx++) {
                        int candidateStatus = NMSCandidateStatus::SELECTED;  // 0 for suppressed, 1 for selected
                        arg.selected_boxes_num = io_selection_size;
                        arg.candidate_box =
                            static_cast<const float*>(&boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num]);
                        arg.candidate_status = static_cast<int*>(&candidateStatus);
                        (*m_jit_kernel)(&arg);
                        if (candidateStatus == NMSCandidateStatus::SELECTED) {
                            boxCoord0[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num];
                            boxCoord1[io_selection_size] =
                                boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num + 1];
                            boxCoord2[io_selection_size] =
                                boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num + 2];
                            boxCoord3[io_selection_size] =
                                boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num + 3];
                            filtBoxes[offset + io_selection_size] = FilteredBox(sorted_boxes[candidate_idx].first,
                                                                                batch_idx,
                                                                                class_idx,
                                                                                sorted_boxes[candidate_idx].second);
                            io_selection_size++;
                        }
                    }
#endif  // OPENVINO_ARCH_X86_64
                } else {
                    for (size_t candidate_idx = 1; (candidate_idx < sortedBoxSize) && (io_selection_size < max_out_box);
                         candidate_idx++) {
                        int candidateStatus = NMSCandidateStatus::SELECTED;  // 0 for suppressed, 1 for selected
                        for (int selected_idx = io_selection_size - 1; selected_idx >= 0; selected_idx--) {
                            float iou = intersectionOverUnion(
                                &boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num],
                                &boxesPtr[filtBoxes[offset + selected_idx].box_index * m_coord_num]);
                            if (iou >= m_iou_threshold) {
                                candidateStatus = NMSCandidateStatus::SUPPRESSED;
                                break;
                            }
                        }

                        if (candidateStatus == NMSCandidateStatus::SELECTED) {
                            filtBoxes[offset + io_selection_size] = FilteredBox(sorted_boxes[candidate_idx].first,
                                                                                batch_idx,
                                                                                class_idx,
                                                                                sorted_boxes[candidate_idx].second);
                            io_selection_size++;
                        }
                    }
                }
            }
        }

        m_num_filtered_boxes[batch_idx][class_idx] = io_selection_size;
    });
}

////////// Rotated boxes //////////

struct RotatedBox {
    float x_ctr, y_ctr, w, h, a;
};

inline float dot_2d(const NonMaxSuppression::Point2D& A, const NonMaxSuppression::Point2D& B) {
    return A.x * B.x + A.y * B.y;
}

inline float cross_2d(const NonMaxSuppression::Point2D& A, const NonMaxSuppression::Point2D& B) {
    return A.x * B.y - B.x * A.y;
}

inline void getRotatedVertices(const float* box, NonMaxSuppression::Point2D (&pts)[4], bool clockwise) {
    auto theta = clockwise ? box[4] : -box[4];

    auto cos_theta = std::cos(theta) * 0.5f;
    auto sin_theta = std::sin(theta) * 0.5f;

    // y: top --> down; x: left --> right
    // Left-Down
    pts[0].x = box[0] - sin_theta * box[3] - cos_theta * box[2];
    pts[0].y = box[1] + cos_theta * box[3] - sin_theta * box[2];
    // Left-Top
    pts[1].x = box[0] + sin_theta * box[3] - cos_theta * box[2];
    pts[1].y = box[1] - cos_theta * box[3] - sin_theta * box[2];
    // Right-Top
    pts[2].x = 2 * box[0] - pts[0].x;
    pts[2].y = 2 * box[1] - pts[0].y;
    // Right-Down
    pts[3].x = 2 * box[0] - pts[1].x;
    pts[3].y = 2 * box[1] - pts[1].y;
}

inline float polygonArea(const NonMaxSuppression::Point2D (&q)[24], const int64_t& m) {
    if (m <= 2l) {
        return 0.f;
    }

    float area = 0.f;
    auto mlu = static_cast<size_t>(m - 1l);
    for (size_t i = 1lu; i < mlu; i++) {
        area += std::abs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
    }

    return area / 2.f;
}

inline size_t convexHullGraham(const NonMaxSuppression::Point2D (&p)[24],
                               const size_t num_in,
                               NonMaxSuppression::Point2D (&q)[24]) {
    OPENVINO_ASSERT(num_in >= 2lu);

    // Step 1:
    // Find point with minimum y
    // if more than 1 points have the same minimum y,
    // pick the one with the minimum x.
    size_t t = 0lu;
    for (size_t i = 1lu; i < num_in; i++) {
        if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
            t = i;
        }
    }
    auto& start = p[t];  // starting point

    // Step 2:
    // Subtract starting point from every points (for sorting in the next step)
    for (size_t i = 0lu; i < num_in; i++) {
        q[i] = p[i] - start;
    }

    // Swap the starting point to position 0
    std::swap(q[t], q[0]);

    // Step 3:
    // Sort point 1 ~ num_in according to their relative cross-product values
    // (essentially sorting according to angles)
    // If the angles are the same, sort according to their distance to origin
    float dist[24];
    for (size_t i = 0lu; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    std::sort(q + 1, q + num_in, [](const NonMaxSuppression::Point2D& A, const NonMaxSuppression::Point2D& B) -> bool {
        float temp = cross_2d(A, B);
        if (std::abs(temp) < 1e-6f) {
            return dot_2d(A, A) < dot_2d(B, B);
        }
        return temp > 0.f;
    });
    // compute distance to origin after sort, since the points are now different.
    for (size_t i = 0lu; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    // Step 4:
    // Make sure there are at least 2 points (that don't overlap with each other)
    // in the stack
    size_t k = 1lu;  // index of the non-overlapped second point
    for (; k < num_in; k++) {
        if (dist[k] > 1e-8f) {
            break;
        }
    }
    if (k == num_in) {
        // We reach the end, which means the convex hull is just one point
        q[0] = p[t];
        return 1lu;
    }
    q[1] = q[k];
    size_t m = 2lu;  // 2 points in the stack
    // Step 5:
    // Finally we can start the scanning process.
    // When a non-convex relationship between the 3 points is found
    // (either concave shape or duplicated points),
    // we pop the previous point from the stack
    // until the 3-point relationship is convex again, or
    // until the stack only contains two points
    for (size_t i = k + 1lu; i < num_in; i++) {
        while (m > 1lu && cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
            m--;
        }
        q[m++] = q[i];
    }

    return m;
}

inline size_t getIntersectionPoints(const NonMaxSuppression::Point2D (&pts1)[4],
                                    const NonMaxSuppression::Point2D (&pts2)[4],
                                    NonMaxSuppression::Point2D (&intersections)[24]) {
    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    NonMaxSuppression::Point2D vec1[4], vec2[4];
    for (size_t i = 0lu; i < 4lu; i++) {
        vec1[i] = pts1[(i + 1lu) % 4lu] - pts1[i];
        vec2[i] = pts2[(i + 1lu) % 4lu] - pts2[i];
    }

    // Line test - test all line combos for intersection
    size_t num = 0lu;  // number of intersections
    for (size_t i = 0lu; i < 4lu; i++) {
        for (size_t j = 0lu; j < 4lu; j++) {
            // Solve for 2x2 Ax=b
            float det = cross_2d(vec2[j], vec1[i]);

            // This takes care of parallel lines
            if (std::abs(det) <= 1e-14f) {
                continue;
            }

            auto vec12 = pts2[j] - pts1[i];

            auto t1 = cross_2d(vec2[j], vec12) / det;
            auto t2 = cross_2d(vec1[i], vec12) / det;

            if (t1 >= 0.f && t1 <= 1.f && t2 >= 0.f && t2 <= 1.f) {
                intersections[num++] = pts1[i] + vec1[i] * t1;
            }
        }
    }

    // Check for vertices of rect1 inside rect2
    {
        const auto& AB = vec2[0];
        const auto& DA = vec2[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (auto i : pts1) {
            // Assume ABCD is the rectangle, and P is the point to be judged
            // P is inside ABCD if P's projection on AB lies within AB
            // and P's projection on AD lies within AD

            auto AP = i - pts2[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = i;
            }
        }
    }

    // Reverse the check - check for vertices of rect2 inside rect1
    {
        const auto& AB = vec1[0];
        const auto& DA = vec1[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (auto i : pts2) {
            auto AP = i - pts1[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = i;
            }
        }
    }

    return num;
}

inline float rotatedBoxesIntersection(const NonMaxSuppression::Point2D (&vertices_0)[4],
                                      const float* box_1,
                                      const bool clockwise) {
    // There are up to 4 x 4 + 4 + 4 = 24 intersections (including duplicates) returned
    NonMaxSuppression::Point2D intersect_pts[24], ordered_pts[24];

    NonMaxSuppression::Point2D vertices_1[4];
    getRotatedVertices(box_1, vertices_1, clockwise);

    auto num = getIntersectionPoints(vertices_0, vertices_1, intersect_pts);

    if (num <= 2lu) {
        return 0.f;
    }

    auto num_convex = convexHullGraham(intersect_pts, num, ordered_pts);
    return polygonArea(ordered_pts, num_convex);
}

inline float NonMaxSuppression::rotatedIntersectionOverUnion(const NonMaxSuppression::Point2D (&vertices_0)[4],
                                                             const float area_0,
                                                             const float* box_1) {
    const auto area_1 = box_1[2] * box_1[3];  // W x H
    if (area_1 <= 0.f) {
        return 0.f;
    }

    const auto intersection = rotatedBoxesIntersection(vertices_0, box_1, m_clockwise);

    return intersection / (area_0 + area_1 - intersection);
}

void NonMaxSuppression::nmsRotated(const float* boxes,
                                   const float* scores,
                                   const VectorDims& boxes_strides,
                                   const VectorDims& scores_strides,
                                   std::vector<FilteredBox>& filtered_boxes) {
    if (m_jit_kernel) {
        THROW_CPU_NODE_ERR("does not have implementation of the JIT kernel for Rotated boxes.");
    } else {
        parallel_for2d(m_batches_num, m_classes_num, [&](int64_t batch_idx, int64_t class_idx) {
            const float* boxes_ptr = boxes + batch_idx * boxes_strides[0];
            const float* scores_ptr = scores + batch_idx * scores_strides[0] + class_idx * scores_strides[1];

            std::vector<std::pair<float, size_t>> sorted_indices;  // score, box_idx
            sorted_indices.reserve(m_boxes_num);
            for (size_t box_idx = 0lu; box_idx < m_boxes_num; box_idx++, scores_ptr++) {
                if (*scores_ptr > m_score_threshold) {
                    sorted_indices.emplace_back(*scores_ptr, box_idx);
                }
            }

            size_t io_selection_size = 0lu;
            const size_t sorted_boxes_size = sorted_indices.size();

            if (sorted_boxes_size > 0lu) {
                parallel_sort(sorted_indices.begin(),
                              sorted_indices.end(),
                              [](const std::pair<float, size_t>& l, const std::pair<float, size_t>& r) {
                                  return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                              });
                auto sorted_indices_ptr = sorted_indices.data();
                auto filtered_boxes_ptr = filtered_boxes.data() + batch_idx * m_classes_num * m_output_boxes_per_class +
                                          class_idx * m_output_boxes_per_class;
                *filtered_boxes_ptr =
                    FilteredBox(sorted_indices[0].first, batch_idx, class_idx, sorted_indices[0].second);
                io_selection_size++;
                if (sorted_boxes_size > 1lu) {
                    sorted_indices_ptr++;
                    NMSCandidateStatus candidate_status;

                    for (size_t candidate_idx = 1lu;
                         (candidate_idx < sorted_boxes_size) && (io_selection_size < m_output_boxes_per_class);
                         candidate_idx++, sorted_indices_ptr++) {
                        candidate_status = NMSCandidateStatus::SELECTED;
                        auto box_0 = boxes_ptr + (*sorted_indices_ptr).second * m_coord_num;
                        const auto area_0 = box_0[2] * box_0[3];  // W x H

                        if (area_0 > 0.f) {
                            NonMaxSuppression::Point2D vertices_0[4];
                            getRotatedVertices(box_0, vertices_0, m_clockwise);
                            auto trg_boxes = reinterpret_cast<int32_t*>(&((*filtered_boxes_ptr).box_index));
                            for (size_t selected_idx = 0lu; selected_idx < io_selection_size;
                                 selected_idx++, trg_boxes -= 4) {
                                auto iou = rotatedIntersectionOverUnion(vertices_0,
                                                                        area_0,
                                                                        boxes_ptr + m_coord_num * (*trg_boxes));
                                if (iou > m_iou_threshold) {
                                    candidate_status = NMSCandidateStatus::SUPPRESSED;
                                    break;
                                }
                            }
                        } else if (0.f > m_iou_threshold) {
                            candidate_status = NMSCandidateStatus::SUPPRESSED;
                        }

                        if (candidate_status == NMSCandidateStatus::SELECTED) {
                            *(++filtered_boxes_ptr) = FilteredBox((*sorted_indices_ptr).first,
                                                                  batch_idx,
                                                                  class_idx,
                                                                  (*sorted_indices_ptr).second);
                            io_selection_size++;
                        }
                    }
                }
            }

            m_num_filtered_boxes[batch_idx][class_idx] = io_selection_size;
        });
    }
}

/////////////// End of Rotated boxes ///////////////

float NonMaxSuppression::intersectionOverUnion(const float* boxesI, const float* boxesJ) {
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    if (boxEncodingType == NMSBoxEncodeType::CENTER) {
        //  box format: x_center, y_center, width, height
        yminI = boxesI[1] - boxesI[3] / 2.f;
        xminI = boxesI[0] - boxesI[2] / 2.f;
        ymaxI = boxesI[1] + boxesI[3] / 2.f;
        xmaxI = boxesI[0] + boxesI[2] / 2.f;
        yminJ = boxesJ[1] - boxesJ[3] / 2.f;
        xminJ = boxesJ[0] - boxesJ[2] / 2.f;
        ymaxJ = boxesJ[1] + boxesJ[3] / 2.f;
        xmaxJ = boxesJ[0] + boxesJ[2] / 2.f;
    } else {
        //  box format: y1, x1, y2, x2
        yminI = (std::min)(boxesI[0], boxesI[2]);
        xminI = (std::min)(boxesI[1], boxesI[3]);
        ymaxI = (std::max)(boxesI[0], boxesI[2]);
        xmaxI = (std::max)(boxesI[1], boxesI[3]);
        yminJ = (std::min)(boxesJ[0], boxesJ[2]);
        xminJ = (std::min)(boxesJ[1], boxesJ[3]);
        ymaxJ = (std::max)(boxesJ[0], boxesJ[2]);
        xmaxJ = (std::max)(boxesJ[1], boxesJ[3]);
    }

    float areaI = (ymaxI - yminI) * (xmaxI - xminI);
    float areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
    if (areaI <= 0.f || areaJ <= 0.f) {
        return 0.f;
    }

    float intersection_area = (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ), 0.f) *
                              (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ), 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
}

void NonMaxSuppression::check1DInput(const Shape& shape, const std::string& name, const size_t port) {
    if (shape.getRank() != 0 && shape.getRank() != 1) {
        THROW_CPU_NODE_ERR("has unsupported '", name, "' input rank: ", shape.getRank());
    }
    if (shape.getRank() == 1) {
        if (shape.getDims()[0] != 1) {
            THROW_CPU_NODE_ERR("has unsupported '", name, "' input 1st dimension size: ", dim2str(shape.getDims()[0]));
        }
    }
}

void NonMaxSuppression::checkOutput(const Shape& shape, const std::string& name, const size_t port) {
    if (shape.getRank() != 2) {
        THROW_CPU_NODE_ERR("has unsupported '", name, "' output rank: ", shape.getRank());
    }
    if (shape.getDims()[1] != 3) {
        THROW_CPU_NODE_ERR("has unsupported '", name, "' output 2nd dimension size: ", dim2str(shape.getDims()[1]));
    }
}

bool NonMaxSuppression::neverExecute() const {
    return !isDynamicNode() && Node::neverExecute();
}

bool NonMaxSuppression::isExecutable() const {
    return isDynamicNode() || Node::isExecutable();
}

bool NonMaxSuppression::created() const {
    return getType() == Type::NonMaxSuppression;
}

}  // namespace ov::intel_cpu::node
