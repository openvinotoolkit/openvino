// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiclass_nms.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/general_utils.h"

using namespace ov;

namespace ov::intel_cpu::node {

using ngNmsSortResultType = ov::op::util::MulticlassNmsBase::SortResultType;

bool MultiClassNms::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v9::MulticlassNms::get_type_info_static(),
                    ov::op::v8::MulticlassNms::get_type_info_static(),
                    ov::op::internal::MulticlassNmsIEInternal::get_type_info_static())) {
            errorMessage = "Node is not an instance of MulticlassNms from opset v8 or v9.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MultiClassNms::MultiClassNms(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (one_of(op->get_type_info(), ov::op::internal::MulticlassNmsIEInternal::get_type_info_static())) {
        m_outStaticShape = true;
    }

    if (getOriginalInputsNumber() != 2 && getOriginalInputsNumber() != 3) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges: ", getOriginalInputsNumber());
    }

    if (getOriginalOutputsNumber() != 3) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges: ", getOriginalOutputsNumber());
    }

    auto nmsBase = ov::as_type_ptr<ov::op::util::MulticlassNmsBase>(op);
    if (nmsBase == nullptr) {
        THROW_CPU_NODE_ERR("is not an instance of MulticlassNmsBase.");
    }
    auto& atrri = nmsBase->get_attrs();
    m_sortResultAcrossBatch = atrri.sort_result_across_batch;
    m_nmsTopK = atrri.nms_top_k;
    m_iouThreshold = atrri.iou_threshold;
    m_scoreThreshold = atrri.score_threshold;
    m_backgroundClass = atrri.background_class;
    m_keepTopK = atrri.keep_top_k;
    if (atrri.sort_result_type == ngNmsSortResultType::CLASSID) {
        m_sortResultType = MulticlassNmsSortResultType::CLASSID;
    } else if (atrri.sort_result_type == ngNmsSortResultType::SCORE) {
        m_sortResultType = MulticlassNmsSortResultType::SCORE;
    } else if (atrri.sort_result_type == ngNmsSortResultType::NONE) {
        m_sortResultType = MulticlassNmsSortResultType::NONE;
    }
    m_nmsEta = atrri.nms_eta;
    m_normalized = atrri.normalized;

    // boxes [N, M, 4], scores [N, C, M] opset8/9
    // boxes [C, M, 4], scores [C, M], roisnum [N] opset9
    const auto& boxes_dims = getInputShapeAtPort(NMS_BOXES).getDims();
    const auto& scores_dims = getInputShapeAtPort(NMS_SCORES).getDims();
    auto boxes_ps = PartialShape(boxes_dims);
    auto scores_ps = PartialShape(scores_dims);
    if (boxes_dims.size() != 3) {
        THROW_CPU_NODE_ERR("has unsupported 'boxes' input rank: ", boxes_dims.size());
    }
    if (boxes_dims[2] != 4) {
        THROW_CPU_NODE_ERR("has unsupported 'boxes' input 3rd dimension size: ", boxes_dims[2]);
    }
    if (scores_dims.size() == 3) {
        if (!boxes_ps[0].compatible(scores_ps[0]) || !boxes_ps[1].compatible(scores_ps[2])) {
            THROW_CPU_NODE_ERR("has incompatible 'boxes' and 'scores' shape ", boxes_ps, " v.s. ", scores_ps);
        }
    } else if (scores_dims.size() == 2) {
        if (op->get_type_info() == ov::op::v8::MulticlassNms::get_type_info_static()) {
            THROW_CPU_NODE_ERR("has unsupported 'scores' input rank: ", scores_dims.size());
        }
        if (!boxes_ps[0].compatible(scores_ps[0]) || !boxes_ps[1].compatible(scores_ps[1])) {
            THROW_CPU_NODE_ERR("has incompatible 'boxes' and 'scores' shape ", boxes_ps, " v.s. ", scores_ps);
        }
        if (getOriginalInputsNumber() != 3) {
            THROW_CPU_NODE_ERR("has incorrect number of input edges: ",
                               getOriginalInputsNumber(),
                               " when input 'scores' is 2D.");
        }
    } else {
        THROW_CPU_NODE_ERR("has unsupported 'scores' input rank: ", scores_dims.size());
    }
}

void MultiClassNms::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    const std::vector<ov::element::Type> supportedFloatPrecision = {ov::element::f32,
                                                                    ov::element::f16,
                                                                    ov::element::bf16};
    const std::vector<ov::element::Type> supportedIntOutputPrecision = {ov::element::i32, ov::element::i64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", m_inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", m_inType);

    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDINDICES),
                   supportedIntOutputPrecision,
                   "selected_indices",
                   m_outType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDOUTPUTS),
                   supportedFloatPrecision,
                   "selected_outputs",
                   m_outType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDNUM),
                   supportedIntOutputPrecision,
                   "selected_num",
                   m_outType);

    if (getOriginalInputsNumber() == 3) {
        checkPrecision(getOriginalInputPrecisionAtPort(NMS_ROISNUM), supportedIntOutputPrecision, "roisnum", m_inType);
        addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::i32}},
                             {{LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32}},
                             impl_desc_type::ref_any);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::f32}},
                             {{LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32}},
                             impl_desc_type::ref_any);
    }
}

// shared           Y               N
// boxes:       N, M, 4         C, sum(M'), 4
// scores:      N, C, M         C, sum(M')
void MultiClassNms::prepareParams() {
    const auto& boxes_dims = getParentEdgeAt(NMS_BOXES)->getMemory().getStaticDims();
    const auto& scores_dims = getParentEdgeAt(NMS_SCORES)->getMemory().getStaticDims();

    const bool has_roinum = getOriginalInputsNumber() == 3;
    const auto shared = scores_dims.size() == 3;  // bboxes shared among classes

    if (shared) {
        if (boxes_dims[0] != scores_dims[0] || boxes_dims[1] != scores_dims[2]) {
            THROW_CPU_NODE_ERR("has incompatible 'boxes' and 'scores' shape ",
                               PartialShape(boxes_dims),
                               " v.s. ",
                               PartialShape(scores_dims));
        }
    } else if (scores_dims.size() == 2) {
        if (boxes_dims[0] != scores_dims[0] || boxes_dims[1] != scores_dims[1]) {
            THROW_CPU_NODE_ERR("has incompatible 'boxes' and 'scores' shape ",
                               PartialShape(boxes_dims),
                               " v.s. ",
                               PartialShape(scores_dims));
        }
        if (!has_roinum) {
            THROW_CPU_NODE_ERR("has incorrect number of input edges: ",
                               getOriginalInputsNumber(),
                               " when input 'scores' is 2D.");
        }
    } else {
        THROW_CPU_NODE_ERR("has unsupported 'scores' input rank: ", scores_dims.size());
    }

    if (has_roinum) {
        const auto& roisnum_dims = getParentEdgeAt(NMS_ROISNUM)->getMemory().getStaticDims();
        if (roisnum_dims.size() != 1) {
            THROW_CPU_NODE_ERR("has unsupported 'roisnum' input rank: ", roisnum_dims.size());
        }
        m_numBatches = shared ? boxes_dims[0] : roisnum_dims[0];
    } else {
        m_numBatches = boxes_dims[0];
    }
    m_numBoxes = boxes_dims[1];
    m_numClasses = shared ? scores_dims[1] : scores_dims[0];

    int max_output_boxes_per_class = 0;
    size_t real_num_classes = m_backgroundClass == -1                                 ? m_numClasses
                              : static_cast<size_t>(m_backgroundClass) < m_numClasses ? m_numClasses - 1
                                                                                      : m_numClasses;
    if (m_nmsTopK) {
        max_output_boxes_per_class = (m_nmsTopK == -1) ? m_numBoxes : std::min(m_nmsTopK, static_cast<int>(m_numBoxes));
        m_filtBoxes.resize(max_output_boxes_per_class * m_numBatches * m_numClasses);
    }
    m_nmsRealTopk = max_output_boxes_per_class;

    m_maxBoxesPerBatch = max_output_boxes_per_class * real_num_classes;
    if (m_keepTopK >= 0) {
        m_maxBoxesPerBatch = std::min(m_maxBoxesPerBatch, static_cast<size_t>(m_keepTopK));
    }

    m_numFiltBox.resize(m_numBatches);  // number of rois after nms for each class in each image
    for (auto& numPerBatch : m_numFiltBox) {
        numPerBatch.resize(m_numClasses, 0);
    }
    m_numBoxOffset.resize(m_numBatches);
}

bool MultiClassNms::neverExecute() const {
    return !isDynamicNode() && Node::neverExecute();
}

bool MultiClassNms::isExecutable() const {
    return isDynamicNode() || Node::isExecutable();
}

void MultiClassNms::executeDynamicImpl(const dnnl::stream& strm) {
    if (hasEmptyInputTensors()) {
        redefineOutputMemory({{0, 6}, {0, 1}, {0}});
        return;
    }
    execute(strm);
}

void MultiClassNms::execute(const dnnl::stream& strm) {
    const auto* boxes = getSrcDataAtPortAs<const float>(NMS_BOXES);
    const auto* scores = getSrcDataAtPortAs<const float>(NMS_SCORES);

    auto dims_boxes = getParentEdgeAt(NMS_BOXES)->getMemory().getStaticDims();
    auto dims_scores = getParentEdgeAt(NMS_SCORES)->getMemory().getStaticDims();

    if (m_nmsRealTopk == 0) {
        return;
    }

    const bool has_roinum = getOriginalInputsNumber() == 3;
    const auto shared = dims_scores.size() == 3;  // bboxes shared among classes

    auto selectedOutputsMemPtr = getDstMemoryAtPort(NMS_SELECTEDOUTPUTS);
    auto selectedIndicesMemPtr = getDstMemoryAtPort(NMS_SELECTEDINDICES);
    auto validOutputsMemPtr = getDstMemoryAtPort(NMS_SELECTEDNUM);

    auto boxesStrides = getParentEdgeAt(NMS_BOXES)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();
    auto scoresStrides = getParentEdgeAt(NMS_SCORES)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();

    int* roisnum = nullptr;
    VectorDims roisnumStrides;
    if (has_roinum) {
        roisnum = getSrcDataAtPortAs<int>(NMS_ROISNUM);
        roisnumStrides = getParentEdgeAt(NMS_ROISNUM)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();
    }

    if ((m_nmsEta >= 0) && (m_nmsEta < 1)) {
        nmsWithEta(boxes, scores, roisnum, boxesStrides, scoresStrides, roisnumStrides, shared);
    } else {
        nmsWithoutEta(boxes, scores, roisnum, boxesStrides, scoresStrides, roisnumStrides, shared);
    }

    size_t startOffset = m_numFiltBox[0][0];
    m_numBoxOffset[0] = 0;
    for (size_t b = 0; b < m_numFiltBox.size(); b++) {
        size_t batchOffsetNew = 0;
        size_t batchOffset = b * m_numClasses * m_nmsRealTopk;
        for (size_t c = (b == 0 ? 1 : 0); c < m_numFiltBox[b].size(); c++) {
            size_t offset = batchOffset + c * m_nmsRealTopk;
            for (size_t i = 0; i < m_numFiltBox[b][c]; i++) {
                m_filtBoxes[startOffset + i] = m_filtBoxes[offset + i];
            }
            startOffset += m_numFiltBox[b][c];
            batchOffsetNew += m_numFiltBox[b][c];
        }
        m_numBoxOffset[b] = batchOffsetNew;
        if (b == 0) {
            m_numBoxOffset[b] += m_numFiltBox[0][0];
        }
    }
    // sort element before go through keep_top_k
    parallel_sort(
        m_filtBoxes.begin(),
        m_filtBoxes.begin() + startOffset,
        [](const filteredBoxes& l, const filteredBoxes& r) {
            return ((l.batch_index < r.batch_index) ||
                    ((l.batch_index == r.batch_index) &&
                     ((l.score > r.score) || ((std::fabs(l.score - r.score) < 1e-6) && l.class_index < r.class_index) ||
                      ((std::fabs(l.score - r.score) < 1e-6) && l.class_index == r.class_index &&
                       l.box_index < r.box_index))));
        });

    if (m_keepTopK > -1) {
        startOffset = 0;
        size_t offset = 0;
        for (size_t b = 0; b < m_numFiltBox.size(); b++) {
            if (m_numBoxOffset[b] > static_cast<size_t>(m_keepTopK)) {
                if (startOffset == offset) {
                    startOffset += m_keepTopK;
                    offset += m_numBoxOffset[b];
                } else {
                    for (int i = 0; i < m_keepTopK; i++) {
                        m_filtBoxes[startOffset + i] = m_filtBoxes[offset + i];
                    }
                    startOffset += m_keepTopK;
                    offset += m_numBoxOffset[b];
                }
            } else {
                if (startOffset == offset) {
                    startOffset += m_numBoxOffset[b];
                    offset += m_numBoxOffset[b];
                } else {
                    for (size_t i = 0; i < m_numBoxOffset[b]; i++) {
                        m_filtBoxes[startOffset + i] = m_filtBoxes[offset + i];
                    }
                    startOffset += m_numBoxOffset[b];
                    offset += m_numBoxOffset[b];
                }
            }
        }
    }

    if (m_sortResultAcrossBatch) { /* sort across batch */
        if (m_sortResultType == MulticlassNmsSortResultType::SCORE) {
            parallel_sort(
                m_filtBoxes.begin(),
                m_filtBoxes.begin() + startOffset,
                [](const filteredBoxes& l, const filteredBoxes& r) {
                    return (l.score > r.score) || (l.score == r.score && l.batch_index < r.batch_index) ||
                           (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                           (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index &&
                            l.box_index < r.box_index);
                });
        } else if (m_sortResultType == MulticlassNmsSortResultType::CLASSID) {
            parallel_sort(
                m_filtBoxes.begin(),
                m_filtBoxes.begin() + startOffset,
                [](const filteredBoxes& l, const filteredBoxes& r) {
                    return (l.class_index < r.class_index) ||
                           (l.class_index == r.class_index && l.batch_index < r.batch_index) ||
                           (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score > r.score) ||
                           (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score == r.score &&
                            l.box_index < r.box_index);
                });
        }
    } else if (m_sortResultType == MulticlassNmsSortResultType::CLASSID) {
        parallel_sort(
            m_filtBoxes.begin(),
            m_filtBoxes.begin() + startOffset,
            [](const filteredBoxes& l, const filteredBoxes& r) {
                return ((l.batch_index < r.batch_index) ||
                        ((l.batch_index == r.batch_index) &&
                         ((l.class_index < r.class_index) || ((l.class_index == r.class_index) && l.score > r.score) ||
                          ((std::fabs(l.score - r.score) <= 1e-6) && l.class_index == r.class_index &&
                           l.box_index < r.box_index))));
            });
    }

    /* output */
    const size_t validOutputs = std::min(startOffset, m_maxBoxesPerBatch * m_numBatches);

    std::vector<size_t> m_selected_num;
    m_selected_num.resize(m_numBatches);

    const size_t selectedBoxesNum_perBatch = m_maxBoxesPerBatch;

    for (size_t idx = 0lu; idx < validOutputs; idx++) {
        m_selected_num[m_filtBoxes[idx].batch_index]++;
    }

    if (!m_outStaticShape) {
        size_t totalBox = std::accumulate(m_selected_num.begin(), m_selected_num.end(), static_cast<size_t>(0));
        redefineOutputMemory({{totalBox, 6}, {totalBox, 1}, {m_numBatches}});
    }
    auto* selected_indices = selectedIndicesMemPtr->getDataAs<int>();
    auto* selected_outputs = selectedOutputsMemPtr->getDataAs<float>();
    auto* selected_num = validOutputsMemPtr->getDataAs<int>();

    auto _flattened_index = [](int batch_idx, int box_idx, int num_box) {
        return batch_idx * num_box + box_idx;
    };

    int64_t output_offset = 0;
    int64_t original_offset = 0;
    for (size_t i = 0; i < m_numBatches; i++) {
        auto real_boxes = m_selected_num[i];
        selected_num[i] = static_cast<int>(real_boxes);

        for (size_t j = 0; j < real_boxes; j++) {
            auto original_index = original_offset + j;
            const auto& box_info = m_filtBoxes[original_index];

            auto selected_base = selected_outputs + (output_offset + j) * 6;
            selected_base[0] = box_info.class_index;
            selected_base[1] = box_info.score;

            auto& selected_index = selected_indices[j + output_offset];
            if (shared) {
                selected_index = _flattened_index(box_info.batch_index, box_info.box_index, m_numBoxes);
                selected_base[2] = boxes[selected_index * 4];
                selected_base[3] = boxes[selected_index * 4 + 1];
                selected_base[4] = boxes[selected_index * 4 + 2];
                selected_base[5] = boxes[selected_index * 4 + 3];
            } else {
                int64_t offset = 0;
                for (int64_t i = 0; i < box_info.batch_index; i++) {
                    offset += roisnum[i];
                }
                // selected index from (M, C, 4)
                selected_index = _flattened_index((offset + box_info.box_index), box_info.class_index, m_numClasses);
                int idx = box_info.class_index * boxesStrides[0] + offset * boxesStrides[1];
                const float* curboxes = boxes + idx;  // a slice of boxes of current class current image
                selected_base[2] = curboxes[4 * box_info.box_index];
                selected_base[3] = curboxes[4 * box_info.box_index + 1];
                selected_base[4] = curboxes[4 * box_info.box_index + 2];
                selected_base[5] = curboxes[4 * box_info.box_index + 3];
            }
        }

        if (m_outStaticShape) {
            std::fill_n(selected_outputs + (output_offset + real_boxes) * 6,
                        (selectedBoxesNum_perBatch - real_boxes) * 6,
                        -1.f);
            std::fill_n(selected_indices + (output_offset + real_boxes), selectedBoxesNum_perBatch - real_boxes, -1);
            output_offset += selectedBoxesNum_perBatch;
            original_offset += real_boxes;
        } else {
            output_offset += real_boxes;
            original_offset += real_boxes;
        }
    }
}

bool MultiClassNms::created() const {
    return getType() == Type::MulticlassNms;
}

float MultiClassNms::intersectionOverUnion(const float* boxesI, const float* boxesJ, const bool normalized) {
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    const auto norm = static_cast<float>(normalized == false);

    // to align with reference
    yminI = boxesI[0];
    xminI = boxesI[1];
    ymaxI = boxesI[2];
    xmaxI = boxesI[3];
    yminJ = boxesJ[0];
    xminJ = boxesJ[1];
    ymaxJ = boxesJ[2];
    xmaxJ = boxesJ[3];

    float areaI = (ymaxI - yminI + norm) * (xmaxI - xminI + norm);
    float areaJ = (ymaxJ - yminJ + norm) * (xmaxJ - xminJ + norm);
    if (areaI <= 0.f || areaJ <= 0.f) {
        return 0.f;
    }

    float intersection_area = (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ) + norm, 0.f) *
                              (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ) + norm, 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
}

void MultiClassNms::nmsWithEta(const float* boxes,
                               const float* scores,
                               const int* roisnum,
                               const VectorDims& boxesStrides,
                               const VectorDims& scoresStrides,
                               const VectorDims& roisnumStrides,
                               const bool shared) {
    auto less = [](const boxInfo& l, const boxInfo& r) {
        return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
    };

    auto func = [](float iou, float adaptive_threshold) {
        return iou <= adaptive_threshold ? 1.0f : 0.0f;
    };

    parallel_for2d(m_numBatches, m_numClasses, [&](int batch_idx, int class_idx) {
        if (!shared) {
            if (roisnum[batch_idx] <= 0) {
                m_numFiltBox[batch_idx][class_idx] = 0;
                return;
            }
        }
        if (class_idx != m_backgroundClass) {
            std::vector<filteredBoxes> fb;
            const float* boxesPtr =
                slice_class(batch_idx, class_idx, boxes, boxesStrides, true, roisnum, roisnumStrides, shared);
            const float* scoresPtr =
                slice_class(batch_idx, class_idx, scores, scoresStrides, false, roisnum, roisnumStrides, shared);

            std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);
            int cur_numBoxes = shared ? m_numBoxes : roisnum[batch_idx];
            for (int box_idx = 0; box_idx < cur_numBoxes; box_idx++) {
                if (scoresPtr[box_idx] >= m_scoreThreshold) {  // algin with ref
                    sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
                }
            }
            fb.reserve(sorted_boxes.size());
            if (sorted_boxes.size() > 0) {
                auto adaptive_threshold = m_iouThreshold;
                int max_out_box =
                    (static_cast<size_t>(m_nmsRealTopk) > sorted_boxes.size()) ? sorted_boxes.size() : m_nmsRealTopk;
                while (max_out_box && !sorted_boxes.empty()) {
                    boxInfo currBox = sorted_boxes.top();
                    float origScore = currBox.score;
                    sorted_boxes.pop();
                    max_out_box--;

                    bool box_is_selected = true;
                    for (int idx = static_cast<int>(fb.size()) - 1; idx >= currBox.suppress_begin_index; idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[currBox.idx * 4],
                                                          &boxesPtr[fb[idx].box_index * 4],
                                                          m_normalized);
                        currBox.score *= func(iou, adaptive_threshold);
                        if (iou >= adaptive_threshold) {
                            box_is_selected = false;
                            break;
                        }
                        if (currBox.score <= m_scoreThreshold) {
                            break;
                        }
                    }

                    currBox.suppress_begin_index = fb.size();
                    if (box_is_selected) {
                        if (m_nmsEta < 1 && adaptive_threshold > 0.5) {
                            adaptive_threshold *= m_nmsEta;
                        }
                        if (currBox.score == origScore) {
                            fb.emplace_back(currBox.score, batch_idx, class_idx, currBox.idx);
                            continue;
                        }
                        if (currBox.score > m_scoreThreshold) {
                            sorted_boxes.push(currBox);
                        }
                    }
                }
            }
            m_numFiltBox[batch_idx][class_idx] = fb.size();
            size_t offset = batch_idx * m_numClasses * m_nmsRealTopk + class_idx * m_nmsRealTopk;
            for (size_t i = 0; i < fb.size(); i++) {
                m_filtBoxes[offset + i] = fb[i];
            }
        }
    });
}

/* get boxes/scores for current class and image
//                  shared         not-shared
// boxes:      [in] N, M, 4         C, M, 4    -> [out] num_priors, 4
// scores:     [in] N, C, M          C, M      -> [out] num_priors,
*/
const float* MultiClassNms::slice_class(const int batch_idx,
                                        const int class_idx,
                                        const float* dataPtr,
                                        const VectorDims& dataStrides,
                                        const bool is_boxes,
                                        const int* roisnum,
                                        const VectorDims& roisnumStrides,
                                        const bool shared) {
    if (shared) {
        if (is_boxes) {
            return dataPtr + batch_idx * dataStrides[0];
        }
        return dataPtr + batch_idx * dataStrides[0] + class_idx * dataStrides[1];
    }

    // get M boxes of current class_idx : 1, M, 4
    const float* boxesPtr_cls = dataPtr + class_idx * dataStrides[0];

    // then get Mi boxes of current batch_idx and current class_idx : M', 4
    auto boxes_idx = 0;
    for (auto i = 0; i < batch_idx; i++) {
        boxes_idx += roisnum[i];
    }
    // auto boxes_num = roisnum[batch_idx];
    return boxesPtr_cls + boxes_idx * dataStrides[1];
}

void MultiClassNms::nmsWithoutEta(const float* boxes,
                                  const float* scores,
                                  const int* roisnum,
                                  const VectorDims& boxesStrides,
                                  const VectorDims& scoresStrides,
                                  const VectorDims& roisnumStrides,
                                  const bool shared) {
    parallel_for2d(m_numBatches, m_numClasses, [&](int batch_idx, int class_idx) {
        /*
        // nms over a class over an image
        // boxes:       num_priors, 4
        // scores:      num_priors, 1
        */
        if (!shared) {
            if (roisnum[batch_idx] <= 0) {
                m_numFiltBox[batch_idx][class_idx] = 0;
                return;
            }
        }
        if (class_idx != m_backgroundClass) {
            const float* boxesPtr =
                slice_class(batch_idx, class_idx, boxes, boxesStrides, true, roisnum, roisnumStrides, shared);
            const float* scoresPtr =
                slice_class(batch_idx, class_idx, scores, scoresStrides, false, roisnum, roisnumStrides, shared);

            std::vector<std::pair<float, int>> sorted_boxes;
            int cur_numBoxes = shared ? m_numBoxes : roisnum[batch_idx];
            for (int box_idx = 0; box_idx < cur_numBoxes; box_idx++) {
                if (scoresPtr[box_idx] >= m_scoreThreshold) {  // align with ref
                    sorted_boxes.emplace_back(scoresPtr[box_idx], box_idx);
                }
            }

            int io_selection_size = 0;
            if (sorted_boxes.size() > 0) {
                parallel_sort(sorted_boxes.begin(),
                              sorted_boxes.end(),
                              [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                                  return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                              });
                int offset = batch_idx * m_numClasses * m_nmsRealTopk + class_idx * m_nmsRealTopk;
                m_filtBoxes[offset + 0] =
                    filteredBoxes(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
                io_selection_size++;
                int max_out_box =
                    (static_cast<size_t>(m_nmsRealTopk) > sorted_boxes.size()) ? sorted_boxes.size() : m_nmsRealTopk;
                for (int box_idx = 1; box_idx < max_out_box; box_idx++) {
                    bool box_is_selected = true;
                    for (int idx = io_selection_size - 1; idx >= 0; idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[sorted_boxes[box_idx].second * 4],
                                                          &boxesPtr[m_filtBoxes[offset + idx].box_index * 4],
                                                          m_normalized);
                        if (iou >= m_iouThreshold) {
                            box_is_selected = false;
                            break;
                        }
                    }

                    if (box_is_selected) {
                        m_filtBoxes[offset + io_selection_size] = filteredBoxes(sorted_boxes[box_idx].first,
                                                                                batch_idx,
                                                                                class_idx,
                                                                                sorted_boxes[box_idx].second);
                        io_selection_size++;
                    }
                }
            }
            m_numFiltBox[batch_idx][class_idx] = io_selection_size;
        }
    });
}

void MultiClassNms::checkPrecision(const ov::element::Type prec,
                                   const std::vector<ov::element::Type>& precList,
                                   const std::string& name,
                                   const std::string& type) {
    if (std::find(precList.begin(), precList.end(), prec) == precList.end()) {
        THROW_CPU_NODE_ERR("has unsupported '", name, "' ", type, " precision: ", prec);
    }
}

}  // namespace ov::intel_cpu::node
