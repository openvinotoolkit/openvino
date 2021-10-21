// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_multiclass_nms.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <ie_ngraph_utils.hpp>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "ie_parallel.hpp"
#include "utils/general_utils.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

using ngNmsSortResultType = ngraph::op::util::NmsBase::SortResultType;

bool MKLDNNMultiClassNmsNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto nms = std::dynamic_pointer_cast<const ngraph::op::v8::MulticlassNms>(op);
        if (!nms) {
            errorMessage = "Only MulticlassNms operation is supported";
            return false;
        }
        const auto& atrri = nms->get_attrs();
        const auto& sortType = atrri.sort_result_type;
        if (!one_of(sortType, ngNmsSortResultType::NONE, ngNmsSortResultType::SCORE, ngNmsSortResultType::CLASSID)) {
            errorMessage = "Does not support SortResultType mode: " + ngraph::as_string(sortType);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMultiClassNmsNode::MKLDNNMultiClassNmsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr& cache)
    : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    m_errorPrefix = "MultiClassNms layer with name '" + getName() + "' ";
    const auto nms = std::dynamic_pointer_cast<const ngraph::op::v8::MulticlassNms>(op);

    auto& atrri = nms->get_attrs();
    m_sort_result_across_batch = atrri.sort_result_across_batch;
    m_nmsTopk = atrri.nms_top_k;
    m_iou_threshold = atrri.iou_threshold;
    m_score_threshold = atrri.score_threshold;
    m_background_class = atrri.background_class;
    m_keep_top_k = atrri.keep_top_k;
    if (atrri.sort_result_type == ngNmsSortResultType::CLASSID)
        m_sort_result_type = MulticlassNmsSortResultType::CLASSID;
    else if (atrri.sort_result_type == ngNmsSortResultType::SCORE)
        m_sort_result_type = MulticlassNmsSortResultType::SCORE;
    else if (atrri.sort_result_type == ngNmsSortResultType::NONE)
        m_sort_result_type = MulticlassNmsSortResultType::NONE;
    m_nms_eta = atrri.nms_eta;
    m_normalized = atrri.normalized;
}

void MKLDNNMultiClassNmsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32, Precision::BF16};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", m_inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", m_inType);

    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDINDICES), supportedIntOutputPrecision, "selected_indices", m_outType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDOUTPUTS), supportedFloatPrecision, "selected_outputs", m_outType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDNUM), supportedIntOutputPrecision, "selected_num", m_outType);

    addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, Precision::FP32}},
                         {{LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, Precision::I32},
                          {LayoutType::ncsp, Precision::I32}},
                         impl_desc_type::ref_any);
}

void MKLDNNMultiClassNmsNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNMultiClassNmsNode::prepareParams() {
    if (!inputShapesDefined()) {
        IE_THROW() << "Can't prepare params for MKLDNNMultiClassNmsNode node with name: " << getName();
    }

    if (getOriginalInputsNumber() != 2)
        IE_THROW() << m_errorPrefix << "has incorrect number of input edges: " << getOriginalInputsNumber();

    if (getOriginalOutputsNumber() != 3)
        IE_THROW() << m_errorPrefix << "has incorrect number of output edges: " << getOriginalOutputsNumber();

    const auto& boxes_dims = getParentEdgeAt(NMS_BOXES)->getMemory().getStaticDims();
    const auto& scores_dims = getParentEdgeAt(NMS_SCORES)->getMemory().getStaticDims();
    if (!(boxes_dims[0] == scores_dims[0] && boxes_dims[1] == scores_dims[2])) {
        IE_THROW() << m_errorPrefix << "has incompatible 'boxes' and 'scores' input dmensions";
    }

    if (boxes_dims.size() != 3)
        IE_THROW() << m_errorPrefix << "has unsupported 'boxes' input rank: " << boxes_dims.size();
    if (boxes_dims[2] != 4)
        IE_THROW() << m_errorPrefix << "has unsupported 'boxes' input 3rd dimension size: " << boxes_dims[2];
    m_numBatches = boxes_dims[0];
    m_numBoxes = boxes_dims[1];

    if (scores_dims.size() != 3)
        IE_THROW() << m_errorPrefix << "has unsupported 'scores' input rank: " << scores_dims.size();
    m_numClasses = scores_dims[1];

    int64_t max_output_boxes_per_class = 0;
    size_t real_num_classes = m_background_class == -1 ? m_numClasses :
        m_background_class < m_numClasses ? m_numClasses - 1 : m_numClasses;
    if (m_nmsTopk >= 0)
        max_output_boxes_per_class = std::min(m_numBoxes, static_cast<size_t>(m_nmsTopk));
    else
        max_output_boxes_per_class = m_numBoxes;

    m_maxBoxesPerBatch = max_output_boxes_per_class * real_num_classes;
    if (m_keep_top_k >= 0)
        m_maxBoxesPerBatch = std::min(m_maxBoxesPerBatch, static_cast<size_t>(m_keep_top_k));

    m_numFiltBox.resize(m_numBatches);
    for (auto &numPerBatch : m_numFiltBox) {
        numPerBatch.resize(m_numClasses, 0);
    }
    m_numBoxOffset.resize(m_numBatches);

    if (m_nmsTopk) {
        m_nmsTopk = (m_nmsTopk == -1) ? m_numBoxes : m_nmsTopk;
        m_filtBoxes.resize(m_nmsTopk * m_numBatches * m_numClasses);
    }
}

void MKLDNNMultiClassNmsNode::execute(mkldnn::stream strm) {
    const float* boxes = reinterpret_cast<const float*>(getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->GetPtr());
    const float* scores = reinterpret_cast<const float*>(getParentEdgeAt(NMS_SCORES)->getMemoryPtr()->GetPtr());

    auto dims_boxes = getParentEdgeAt(NMS_BOXES)->getMemory().getStaticDims();

    if (m_nmsTopk == 0)
        return;

    auto selectedOutputsMemPtr = getChildEdgesAtPort(NMS_SELECTEDOUTPUTS)[0]->getMemoryPtr();
    auto selectedIndicesMemPtr = getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getMemoryPtr();
    auto validOutputsMemPtr = getChildEdgesAtPort(NMS_SELECTEDNUM)[0]->getMemoryPtr();

    auto boxesStrides = getParentEdgeAt(NMS_BOXES)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    auto scoresStrides = getParentEdgeAt(NMS_SCORES)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();

    if ((m_nms_eta >= 0) && (m_nms_eta < 1)) {
        nmsWithEta(boxes, scores, boxesStrides, scoresStrides);
    } else {
        nmsWithoutEta(boxes, scores, boxesStrides, scoresStrides);
    }

    size_t startOffset = m_numFiltBox[0][0];
    m_numBoxOffset[0] = 0;
    for (size_t b = 0; b < m_numFiltBox.size(); b++) {
        size_t batchOffsetNew = 0;
        size_t batchOffset = b * m_numClasses * m_nmsTopk;
        for (size_t c = (b == 0 ? 1 : 0); c < m_numFiltBox[b].size(); c++) {
            size_t offset = batchOffset + c * m_nmsTopk;
            for (size_t i = 0; i < m_numFiltBox[b][c]; i++) {
                m_filtBoxes[startOffset + i] = m_filtBoxes[offset + i];
            }
            startOffset += m_numFiltBox[b][c];
            batchOffsetNew += m_numFiltBox[b][c];
        }
        m_numBoxOffset[b] = batchOffsetNew;
        if (b == 0)
            m_numBoxOffset[b] += m_numFiltBox[0][0];
    }
    // sort element before go through keep_top_k
    parallel_sort(m_filtBoxes.begin(), m_filtBoxes.begin() + startOffset, [](const filteredBoxes& l, const filteredBoxes& r) {
        return ((l.batch_index < r.batch_index) ||
                ((l.batch_index == r.batch_index) && ((l.score > r.score) || ((std::fabs(l.score - r.score) < 1e-6) && l.class_index < r.class_index) ||
                                                      ((std::fabs(l.score - r.score) < 1e-6) && l.class_index == r.class_index && l.box_index < r.box_index))));
    });

    if (m_keep_top_k > -1) {
        startOffset = 0;
        size_t offset = 0;
        for (size_t b = 0; b < m_numFiltBox.size(); b++) {
            if (m_numBoxOffset[b] > m_keep_top_k) {
                if (startOffset == offset) {
                    startOffset += m_keep_top_k;
                    offset += m_numBoxOffset[b];
                } else {
                    for (size_t i = 0; i < m_keep_top_k; i++) {
                        m_filtBoxes[startOffset + i] = m_filtBoxes[offset + i];
                    }
                    startOffset += m_keep_top_k;
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

    if (m_sort_result_across_batch) {
        if (m_sort_result_type == SCORE) {
            parallel_sort(m_filtBoxes.begin(), m_filtBoxes.begin() + startOffset, [](const filteredBoxes& l, const filteredBoxes& r) {
                return (l.score > r.score) || (l.score == r.score && l.batch_index < r.batch_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index && l.box_index < r.box_index);
            });
        } else if (m_sort_result_type == CLASSID) {
            parallel_sort(m_filtBoxes.begin(), m_filtBoxes.begin() + startOffset, [](const filteredBoxes& l, const filteredBoxes& r) {
                return (l.class_index < r.class_index) || (l.class_index == r.class_index && l.batch_index < r.batch_index) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score > r.score) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score == r.score && l.box_index < r.box_index);
            });
        }
    } else if (m_sort_result_type == CLASSID) {
        parallel_sort(m_filtBoxes.begin(), m_filtBoxes.begin() + startOffset, [](const filteredBoxes& l, const filteredBoxes& r) {
            return ((l.batch_index < r.batch_index) ||
                    ((l.batch_index == r.batch_index) &&
                     ((l.class_index < r.class_index) || ((l.class_index == r.class_index) && l.score > r.score) ||
                      ((std::fabs(l.score - r.score) <= 1e-6) && l.class_index == r.class_index && l.box_index < r.box_index))));
        });
    }

    const size_t validOutputs = std::min(startOffset, m_maxBoxesPerBatch * dims_boxes[0]);

    std::vector<size_t> m_selected_num;
    m_selected_num.resize(dims_boxes[0]);

    const size_t selectedBoxesNum_perBatch = m_maxBoxesPerBatch;

    for (size_t idx = 0lu; idx < validOutputs; idx++) {
        m_selected_num[m_filtBoxes[idx].batch_index]++;
    }

    if (isDynamicNode()) {
        size_t totalBox = std::accumulate(m_selected_num.begin(), m_selected_num.end(), 0);
        selectedOutputsMemPtr->redefineDesc(getBaseMemDescAtOutputPort(NMS_SELECTEDOUTPUTS)->cloneWithNewDims({totalBox, 6}));
        selectedIndicesMemPtr->redefineDesc(getBaseMemDescAtOutputPort(NMS_SELECTEDINDICES)->cloneWithNewDims({totalBox, 1}));
        validOutputsMemPtr->redefineDesc(getBaseMemDescAtOutputPort(NMS_SELECTEDNUM)->cloneWithNewDims({m_numBatches}));
    }
    int* selected_indices = reinterpret_cast<int*>(selectedIndicesMemPtr->GetPtr());
    float* selected_outputs = reinterpret_cast<float*>(selectedOutputsMemPtr->GetPtr());
    int* selected_num = reinterpret_cast<int*>(validOutputsMemPtr->GetPtr());

    int64_t output_offset = 0;
    int64_t original_offset = 0;
    for (size_t i = 0; i < dims_boxes[0]; i++) {
        auto real_boxes = m_selected_num[i];
        selected_num[i] = static_cast<int>(real_boxes);

        for (size_t j = 0; j < real_boxes; j++) {
            auto original_index = original_offset + j;
            selected_indices[j + output_offset] = m_filtBoxes[original_index].batch_index * dims_boxes[1] + m_filtBoxes[original_index].box_index;
            auto selected_base = selected_outputs + (output_offset + j) * 6;
            selected_base[0] = m_filtBoxes[original_index].class_index;
            selected_base[1] = m_filtBoxes[original_index].score;
            selected_base[2] = boxes[selected_indices[j + output_offset] * 4];
            selected_base[3] = boxes[selected_indices[j + output_offset] * 4 + 1];
            selected_base[4] = boxes[selected_indices[j + output_offset] * 4 + 2];
            selected_base[5] = boxes[selected_indices[j + output_offset] * 4 + 3];
        }
        if (!isDynamicNode()) {
            std::fill_n(selected_outputs + (output_offset + real_boxes) * 6, (selectedBoxesNum_perBatch - real_boxes) * 6, -1);
            std::fill_n(selected_indices + (output_offset + real_boxes), selectedBoxesNum_perBatch - real_boxes, -1);
            output_offset += selectedBoxesNum_perBatch;
            original_offset += real_boxes;
        } else {
            output_offset += real_boxes;
            original_offset += real_boxes;
        }
    }
}

bool MKLDNNMultiClassNmsNode::created() const {
    return getType() == MulticlassNms;
}

float MKLDNNMultiClassNmsNode::intersectionOverUnion(const float* boxesI, const float* boxesJ, const bool normalized) {
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    const float norm = static_cast<float>(normalized == false);

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
    if (areaI <= 0.f || areaJ <= 0.f)
        return 0.f;

    float intersection_area = (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ) + norm, 0.f) *
                              (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ) + norm, 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
}

void MKLDNNMultiClassNmsNode::nmsWithEta(const float* boxes, const float* scores, const SizeVector& boxesStrides, const SizeVector& scoresStrides) {
    auto less = [](const boxInfo& l, const boxInfo& r) {
        return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
    };

    auto func = [](float iou, float adaptive_threshold) {
        return iou <= adaptive_threshold ? 1.0f : 0.0f;
    };

    parallel_for2d(m_numBatches, m_numClasses, [&](int batch_idx, int class_idx) {
        if (class_idx != m_background_class) {
            std::vector<filteredBoxes> fb;
            const float* boxesPtr = boxes + batch_idx * boxesStrides[0];
            const float* scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

            std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);
            for (int box_idx = 0; box_idx < m_numBoxes; box_idx++) {
                if (scoresPtr[box_idx] >= m_score_threshold)  // algin with ref
                    sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
            }
            fb.reserve(sorted_boxes.size());
            if (sorted_boxes.size() > 0) {
                auto adaptive_threshold = m_iou_threshold;
                int max_out_box = (m_nmsTopk > sorted_boxes.size()) ? sorted_boxes.size() : m_nmsTopk;
                while (max_out_box && !sorted_boxes.empty()) {
                    boxInfo currBox = sorted_boxes.top();
                    float origScore = currBox.score;
                    sorted_boxes.pop();
                    max_out_box--;

                    bool box_is_selected = true;
                    for (int idx = static_cast<int>(fb.size()) - 1; idx >= currBox.suppress_begin_index; idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[currBox.idx * 4], &boxesPtr[fb[idx].box_index * 4], m_normalized);
                        currBox.score *= func(iou, adaptive_threshold);
                        if (iou >= adaptive_threshold) {
                            box_is_selected = false;
                            break;
                        }
                        if (currBox.score <= m_score_threshold)
                            break;
                    }

                    currBox.suppress_begin_index = fb.size();
                    if (box_is_selected) {
                        if (m_nms_eta < 1 && adaptive_threshold > 0.5) {
                            adaptive_threshold *= m_nms_eta;
                        }
                        if (currBox.score == origScore) {
                            fb.push_back({currBox.score, batch_idx, class_idx, currBox.idx});
                            continue;
                        }
                        if (currBox.score > m_score_threshold) {
                            sorted_boxes.push(currBox);
                        }
                    }
                }
            }
            m_numFiltBox[batch_idx][class_idx] = fb.size();
            size_t offset = batch_idx * m_numClasses * m_nmsTopk + class_idx * m_nmsTopk;
            for (size_t i = 0; i < fb.size(); i++) {
                m_filtBoxes[offset + i] = fb[i];
            }
        }
    });
}

void MKLDNNMultiClassNmsNode::nmsWithoutEta(const float* boxes, const float* scores, const SizeVector& boxesStrides, const SizeVector& scoresStrides) {
    parallel_for2d(m_numBatches, m_numClasses, [&](int batch_idx, int class_idx) {
        if (class_idx != m_background_class) {
            const float* boxesPtr = boxes + batch_idx * boxesStrides[0];
            const float* scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

            std::vector<std::pair<float, int>> sorted_boxes;
            for (int box_idx = 0; box_idx < m_numBoxes; box_idx++) {
                if (scoresPtr[box_idx] >= m_score_threshold)  // algin with ref
                    sorted_boxes.emplace_back(std::make_pair(scoresPtr[box_idx], box_idx));
            }

            int io_selection_size = 0;
            if (sorted_boxes.size() > 0) {
                parallel_sort(sorted_boxes.begin(), sorted_boxes.end(), [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                    return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                });
                int offset = batch_idx * m_numClasses * m_nmsTopk + class_idx * m_nmsTopk;
                m_filtBoxes[offset + 0] = filteredBoxes(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
                io_selection_size++;
                int max_out_box = (m_nmsTopk > sorted_boxes.size()) ? sorted_boxes.size() : m_nmsTopk;
                for (size_t box_idx = 1; box_idx < max_out_box; box_idx++) {
                    bool box_is_selected = true;
                    for (int idx = io_selection_size - 1; idx >= 0; idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[sorted_boxes[box_idx].second * 4],
                            &boxesPtr[m_filtBoxes[offset + idx].box_index * 4], m_normalized);
                        if (iou >= m_iou_threshold) {
                            box_is_selected = false;
                            break;
                        }
                    }

                    if (box_is_selected) {
                        m_filtBoxes[offset + io_selection_size] = filteredBoxes(sorted_boxes[box_idx].first, batch_idx, class_idx,
                            sorted_boxes[box_idx].second);
                        io_selection_size++;
                    }
                }
            }
            m_numFiltBox[batch_idx][class_idx] = io_selection_size;
        }
    });
}

void MKLDNNMultiClassNmsNode::checkPrecision(const Precision prec, const std::vector<Precision> precList, const std::string name, const std::string type) {
    if (std::find(precList.begin(), precList.end(), prec) == precList.end())
        IE_THROW() << m_errorPrefix << "has unsupported '" << name << "' " << type << " precision: " << prec;
}

REG_MKLDNN_PRIM_FOR(MKLDNNMultiClassNmsNode, MulticlassNms)