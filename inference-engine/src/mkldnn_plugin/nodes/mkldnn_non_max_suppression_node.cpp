// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <queue>

#include "mkldnn_non_max_suppression_node.h"
#include "ie_parallel.hpp"
#include <ngraph_ops/nms_ie_internal.hpp>
#include "utils/general_utils.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNNonMaxSuppressionNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto nms = std::dynamic_pointer_cast<const ngraph::op::internal::NonMaxSuppressionIEInternal>(op);
        if (!nms) {
            errorMessage = "Only internal NonMaxSuppression operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNNonMaxSuppressionNode::MKLDNNNonMaxSuppressionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
        std::string errorMessage;
        if (!isSupportedOperation(op, errorMessage)) {
            IE_THROW(NotImplemented) << errorMessage;
        }

        errorPrefix = "NMS layer with name '" + op->get_friendly_name() + "' ";
        const auto nms = std::dynamic_pointer_cast<const ngraph::op::internal::NonMaxSuppressionIEInternal>(op);

        if (getOriginalInputsNumber() < 2 || getOriginalInputsNumber() > 6)
            IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getOriginalInputsNumber();

        if (getOriginalOutputsNumber() < 1 || getOriginalOutputsNumber() > 3)
            IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getOriginalOutputsNumber();

        boxEncodingType = nms->m_center_point_box ? boxEncoding::CENTER : boxEncoding::CORNER;

        sort_result_descending = nms->m_sort_result_descending;

        const SizeVector &boxes_dims = op->get_input_shape(NMS_BOXES);
        num_batches = boxes_dims[0];
        num_boxes = boxes_dims[1];
        if (boxes_dims.size() != 3)
            IE_THROW() << errorPrefix << "has unsupported 'boxes' input rank: " << boxes_dims.size();
        if (boxes_dims[2] != 4)
            IE_THROW() << errorPrefix << "has unsupported 'boxes' input 3rd dimension size: " << boxes_dims[2];

        const SizeVector &scores_dims = op->get_input_shape(NMS_SCORES);
        num_classes = scores_dims[1];
        if (scores_dims.size() != 3)
            IE_THROW() << errorPrefix << "has unsupported 'scores' input rank: " << scores_dims.size();

        if (num_batches != scores_dims[0])
            IE_THROW() << errorPrefix << " num_batches is different in 'boxes' and 'scores' inputs";
        if (num_boxes != scores_dims[2])
            IE_THROW() << errorPrefix << " num_boxes is different in 'boxes' and 'scores' inputs";

        numFiltBox.resize(num_batches);
        for (auto & i : numFiltBox)
            i.resize(num_classes);

        inputShape_MAXOUTPUTBOXESPERCLASS = op->get_input_shape(NMS_MAXOUTPUTBOXESPERCLASS);
        inputShape_IOUTHRESHOLD = op->get_input_shape(NMS_IOUTHRESHOLD);
        inputShape_SCORETHRESHOLD = op->get_input_shape(NMS_SCORETHRESHOLD);
        if (getOriginalInputsNumber() > NMS_SOFTNMSSIGMA) {
            inputShape_SOFTNMSSIGMA = op->get_input_shape(NMS_SOFTNMSSIGMA);
        }

        outputShape_SELECTEDINDICES = op->get_output_shape(NMS_SELECTEDINDICES);
        outputShape_SELECTEDSCORES = op->get_output_shape(NMS_SELECTEDSCORES);

        const SizeVector &valid_outputs_dims = op->get_input_shape(NMS_VALIDOUTPUTS);
        if (valid_outputs_dims.size() != 1)
            IE_THROW() << errorPrefix << "has unsupported 'valid_outputs' output rank: " << valid_outputs_dims.size();
        if (valid_outputs_dims[0] != 1)
            IE_THROW() << errorPrefix << "has unsupported 'valid_outputs' output 1st dimension size: " << valid_outputs_dims[1];
}

void MKLDNNNonMaxSuppressionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32, Precision::BF16};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_VALIDOUTPUTS), supportedIntOutputPrecision, "valid_outputs", outType);

    const std::vector<Precision> supportedPrecision = {Precision::I16, Precision::U8, Precision::I8, Precision::U16, Precision::I32,
                                                       Precision::U32, Precision::I64, Precision::U64};

    check1DInput(inputShape_MAXOUTPUTBOXESPERCLASS, supportedPrecision, "max_output_boxes_per_class", NMS_MAXOUTPUTBOXESPERCLASS);
    check1DInput(inputShape_IOUTHRESHOLD, supportedFloatPrecision, "iou_threshold", NMS_IOUTHRESHOLD);
    check1DInput(inputShape_SCORETHRESHOLD, supportedFloatPrecision, "score_threshold", NMS_SCORETHRESHOLD);

    if (getOriginalInputsNumber() > NMS_SOFTNMSSIGMA) {
        check1DInput(inputShape_SOFTNMSSIGMA, supportedFloatPrecision, "soft_nms_sigma", NMS_SOFTNMSSIGMA);
    }

    checkOutput(outputShape_SELECTEDINDICES, supportedIntOutputPrecision, "selected_indices", NMS_SELECTEDINDICES);
    checkOutput(outputShape_SELECTEDSCORES, supportedFloatPrecision, "selected_scores", NMS_SELECTEDSCORES);

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(getOriginalInputsNumber());
    for (int i = 0; i < getOriginalInputsNumber(); ++i) {
        Precision inPrecision = i == NMS_MAXOUTPUTBOXESPERCLASS ? Precision::I32 : Precision::FP32;
        inDataConf.emplace_back(GeneralLayout::ncsp, inPrecision);
    }

    std::vector<PortConfigurator> outDataConf;
    outDataConf.reserve(getOriginalOutputsNumber());
    for (int i = 0; i < getOriginalOutputsNumber(); ++i) {
        Precision outPrecision = i == NMS_SELECTEDSCORES ? Precision::FP32 : Precision::I32;
        outDataConf.emplace_back(GeneralLayout::ncsp, outPrecision);
    }

    addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
}

void MKLDNNNonMaxSuppressionNode::execute(mkldnn::stream strm) {
    const float *boxes = reinterpret_cast<const float *>(getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->GetPtr());
    const float *scores = reinterpret_cast<const float *>(getParentEdgeAt(NMS_SCORES)->getMemoryPtr()->GetPtr());

    max_output_boxes_per_class = outputShapes.size() > NMS_SELECTEDSCORES ? 0 : num_boxes;
    if (inputShapes.size() > NMS_MAXOUTPUTBOXESPERCLASS) {
        max_output_boxes_per_class = reinterpret_cast<int *>(getParentEdgeAt(NMS_MAXOUTPUTBOXESPERCLASS)->getMemoryPtr()->GetPtr())[0];
    }

    if (max_output_boxes_per_class == 0)
        return;

    iou_threshold = outputShapes.size() > NMS_SELECTEDSCORES ? 0.0f : 1.0f;
    if (inputShapes.size() > NMS_IOUTHRESHOLD)
        iou_threshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_IOUTHRESHOLD)->getMemoryPtr()->GetPtr())[0];

    score_threshold = 0.0f;
    if (inputShapes.size() > NMS_SCORETHRESHOLD)
        score_threshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_SCORETHRESHOLD)->getMemoryPtr()->GetPtr())[0];

    soft_nms_sigma = 0.0f;
    if (inputShapes.size() > NMS_SOFTNMSSIGMA)
        soft_nms_sigma = reinterpret_cast<float *>(getParentEdgeAt(NMS_SOFTNMSSIGMA)->getMemoryPtr()->GetPtr())[0];
    scale = 0.0f;
    if (soft_nms_sigma > 0.0) {
        scale = -0.5 / soft_nms_sigma;
    }

    int *selected_indices = reinterpret_cast<int *>(getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getMemoryPtr()->GetPtr());

    float *selected_scores = nullptr;
    if (outputShapes.size() > NMS_SELECTEDSCORES)
        selected_scores = reinterpret_cast<float *>(getChildEdgesAtPort(NMS_SELECTEDSCORES)[0]->getMemoryPtr()->GetPtr());

    int *valid_outputs = nullptr;
    if (outputShapes.size() > NMS_VALIDOUTPUTS)
        valid_outputs = reinterpret_cast<int *>(getChildEdgesAtPort(NMS_VALIDOUTPUTS)[0]->getMemoryPtr()->GetPtr());

    auto boxesStrides = getParentEdgeAt(NMS_BOXES)->getMemory().GetDescWithType<BlockedMemoryDesc>().getStrides();
    auto scoresStrides = getParentEdgeAt(NMS_SCORES)->getMemory().GetDescWithType<BlockedMemoryDesc>().getStrides();

    std::vector<filteredBoxes> filtBoxes(max_output_boxes_per_class * num_batches * num_classes);

    if (soft_nms_sigma == 0.0f) {
        nmsWithoutSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
    } else {
        nmsWithSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
    }

    size_t startOffset = numFiltBox[0][0];
    for (size_t b = 0; b < numFiltBox.size(); b++) {
        size_t batchOffset = b*num_classes*max_output_boxes_per_class;
        for (size_t c = (b == 0 ? 1 : 0); c < numFiltBox[b].size(); c++) {
            size_t offset = batchOffset + c*max_output_boxes_per_class;
            for (size_t i = 0; i < numFiltBox[b][c]; i++) {
                filtBoxes[startOffset + i] = filtBoxes[offset + i];
            }
            startOffset += numFiltBox[b][c];
        }
    }
    filtBoxes.resize(startOffset);

    // need more particular comparator to get deterministic behaviour
    // escape situation when filtred boxes with same score have different position from launch to launch
    if (sort_result_descending) {
        parallel_sort(filtBoxes.begin(), filtBoxes.end(),
                      [](const filteredBoxes& l, const filteredBoxes& r) {
                          return (l.score > r.score) ||
                                 (l.score ==  r.score && l.batch_index < r.batch_index) ||
                                 (l.score ==  r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                                 (l.score ==  r.score && l.batch_index == r.batch_index && l.class_index == r.class_index && l.box_index < r.box_index);
                      });
    }

    const size_t selectedBoxesNum = getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getShape().getStaticDims()[0];
    const size_t validOutputs = std::min(filtBoxes.size(), selectedBoxesNum);

    int selectedIndicesStride = getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getMemory().GetDescWithType<BlockedMemoryDesc>().getStrides()[0];
    int *selectedIndicesPtr = selected_indices;
    float *selectedScoresPtr = selected_scores;

    size_t idx = 0lu;
    for (; idx < validOutputs; idx++) {
        selectedIndicesPtr[0] = filtBoxes[idx].batch_index;
        selectedIndicesPtr[1] = filtBoxes[idx].class_index;
        selectedIndicesPtr[2] = filtBoxes[idx].box_index;
        selectedIndicesPtr += selectedIndicesStride;
        if (outputShapes.size() > NMS_SELECTEDSCORES) {
            selectedScoresPtr[0] = static_cast<float>(filtBoxes[idx].batch_index);
            selectedScoresPtr[1] = static_cast<float>(filtBoxes[idx].class_index);
            selectedScoresPtr[2] = static_cast<float>(filtBoxes[idx].score);
            selectedScoresPtr += selectedIndicesStride;
        }
    }
    std::fill(selectedIndicesPtr, selectedIndicesPtr + (selectedBoxesNum - idx) * selectedIndicesStride, -1);
    if (outputShapes.size() > NMS_SELECTEDSCORES) {
        std::fill(selectedScoresPtr, selectedScoresPtr + (selectedBoxesNum - idx) * selectedIndicesStride, -1.f);
    }
    if (outputShapes.size() > NMS_VALIDOUTPUTS)
        *valid_outputs = static_cast<int>(validOutputs);
}

bool MKLDNNNonMaxSuppressionNode::created() const {
    return getType() == NonMaxSuppression;
}

float MKLDNNNonMaxSuppressionNode::intersectionOverUnion(const float *boxesI, const float *boxesJ) {
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    if (boxEncodingType == boxEncoding::CENTER) {
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
    if (areaI <= 0.f || areaJ <= 0.f)
        return 0.f;

    float intersection_area =
            (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ), 0.f) *
            (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ), 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
}

void MKLDNNNonMaxSuppressionNode::nmsWithSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides,
                                                             const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
    auto less = [](const boxInfo& l, const boxInfo& r) {
        return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
    };

    auto coeff = [&](float iou) {
        const float weight = std::exp(scale * iou * iou);
        return iou <= iou_threshold ? weight : 0.0f;
    };

    parallel_for2d(num_batches, num_classes, [&](int batch_idx, int class_idx) {
        std::vector<filteredBoxes> fb;
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);
        for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
            if (scoresPtr[box_idx] > score_threshold)
                sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
        }

        fb.reserve(sorted_boxes.size());
        if (sorted_boxes.size() > 0) {
            while (fb.size() < max_output_boxes_per_class && !sorted_boxes.empty()) {
                boxInfo currBox = sorted_boxes.top();
                float origScore = currBox.score;
                sorted_boxes.pop();

                bool box_is_selected = true;
                for (int idx = static_cast<int>(fb.size()) - 1; idx >= currBox.suppress_begin_index; idx--) {
                    float iou = intersectionOverUnion(&boxesPtr[currBox.idx * 4], &boxesPtr[fb[idx].box_index * 4]);
                    currBox.score *= coeff(iou);
                    if (iou >= iou_threshold) {
                        box_is_selected = false;
                        break;
                    }
                    if (currBox.score <= score_threshold)
                        break;
                }

                currBox.suppress_begin_index = fb.size();
                if (box_is_selected) {
                    if (currBox.score == origScore) {
                        fb.push_back({ currBox.score, batch_idx, class_idx, currBox.idx });
                        continue;
                    }
                    if (currBox.score > score_threshold) {
                        sorted_boxes.push(currBox);
                    }
                }
            }
        }
        numFiltBox[batch_idx][class_idx] = fb.size();
        size_t offset = batch_idx*num_classes*max_output_boxes_per_class + class_idx*max_output_boxes_per_class;
        for (size_t i = 0; i < fb.size(); i++) {
            filtBoxes[offset + i] = fb[i];
        }
    });
}

void MKLDNNNonMaxSuppressionNode::nmsWithoutSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides,
                                                                const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
    int max_out_box = static_cast<int>(max_output_boxes_per_class);
    parallel_for2d(num_batches, num_classes, [&](int batch_idx, int class_idx) {
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::vector<std::pair<float, int>> sorted_boxes;
        for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
            if (scoresPtr[box_idx] > score_threshold)
                sorted_boxes.emplace_back(std::make_pair(scoresPtr[box_idx], box_idx));
        }

        int io_selection_size = 0;
        if (sorted_boxes.size() > 0) {
            parallel_sort(sorted_boxes.begin(), sorted_boxes.end(),
                          [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                              return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                          });
            int offset = batch_idx*num_classes*max_output_boxes_per_class + class_idx*max_output_boxes_per_class;
            filtBoxes[offset + 0] = filteredBoxes(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
            io_selection_size++;
            for (size_t box_idx = 1; (box_idx < sorted_boxes.size()) && (io_selection_size < max_out_box); box_idx++) {
                bool box_is_selected = true;
                for (int idx = io_selection_size - 1; idx >= 0; idx--) {
                    float iou = intersectionOverUnion(&boxesPtr[sorted_boxes[box_idx].second * 4], &boxesPtr[filtBoxes[offset + idx].box_index * 4]);
                    if (iou >= iou_threshold) {
                        box_is_selected = false;
                        break;
                    }
                }

                if (box_is_selected) {
                    filtBoxes[offset + io_selection_size] = filteredBoxes(sorted_boxes[box_idx].first, batch_idx, class_idx, sorted_boxes[box_idx].second);
                    io_selection_size++;
                }
            }
        }
        numFiltBox[batch_idx][class_idx] = io_selection_size;
    });
}

void MKLDNNNonMaxSuppressionNode::checkPrecision(const Precision prec, const std::vector<Precision> precList,
                                                           const std::string name, const std::string type) {
    if (std::find(precList.begin(), precList.end(), prec) == precList.end())
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' " << type << " precision: " << prec;
}

void MKLDNNNonMaxSuppressionNode::check1DInput(const SizeVector& dims, const std::vector<Precision> precList,
                                                         const std::string name, const size_t port) {
    checkPrecision(getOriginalInputPrecisionAtPort(port), precList, name, inType);

    if (dims.size() != 0 && dims.size() != 1)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' input rank: " << dims.size();
    if (dims.size() == 1)
        if (dims[0] != 1)
            IE_THROW() << errorPrefix << "has unsupported '" << name << "' input 1st dimension size: " << dims[0];
}

void MKLDNNNonMaxSuppressionNode::checkOutput(const SizeVector& dims, const std::vector<Precision> precList,
                                                        const std::string name, const size_t port) {
    checkPrecision(getOriginalOutputPrecisionAtPort(port), precList, name, outType);

    if (dims.size() != 2)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' output rank: " << dims.size();
    if (dims[1] != 3)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' output 2nd dimension size: " << dims[1];
}


REG_MKLDNN_PRIM_FOR(MKLDNNNonMaxSuppressionNode, NonMaxSuppression)
