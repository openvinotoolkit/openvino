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
#include <ngraph/opsets/opset5.hpp>
#include <ngraph_ops/nms_ie_internal.hpp>
#include "utils/general_utils.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNNonMaxSuppressionNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        // TODO [DS NMS]: remove when nodes from models where nms is not last node in model supports DS
        using NonMaxSuppressionV5 = ngraph::op::v5::NonMaxSuppression;
        if (!one_of(op->get_type_info(), NonMaxSuppressionV5::get_type_info_static(),
                    ngraph::op::internal::NonMaxSuppressionIEInternal::get_type_info_static())) {
            errorMessage = "Only NonMaxSuppression v5 and NonMaxSuppressionIEInternal are supported";
            return false;
        }

        if (const auto nms5 = std::dynamic_pointer_cast<const NonMaxSuppressionV5>(op)) {
            const auto boxEncoding = nms5->get_box_encoding();
            if (!one_of(boxEncoding, NonMaxSuppressionV5::BoxEncodingType::CENTER, NonMaxSuppressionV5::BoxEncodingType::CORNER)) {
                errorMessage = "Supports only CENTER and CORNER box encoding type";
                return false;
            }
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

        if (getOriginalInputsNumber() < 2 || getOriginalInputsNumber() > 6)
            IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getOriginalInputsNumber();

        if (getOriginalOutputsNumber() != 3)
            IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getOriginalOutputsNumber();

        if (const auto nms5 = std::dynamic_pointer_cast<const ngraph::op::v5::NonMaxSuppression>(op)) {
            boxEncodingType = static_cast<boxEncoding>(nms5->get_box_encoding());
            sort_result_descending = nms5->get_sort_result_descending();
        // TODO [DS NMS]: remove when nodes from models where nms is not last node in model supports DS
        } else if (const auto nmsIe = std::dynamic_pointer_cast<const ngraph::op::internal::NonMaxSuppressionIEInternal>(op)) {
            boxEncodingType = nmsIe->m_center_point_box ? boxEncoding::CENTER : boxEncoding::CORNER;
            sort_result_descending = nmsIe->m_sort_result_descending;
        } else {
            const auto &typeInfo = op->get_type_info();
            IE_THROW() << errorPrefix << " doesn't support NMS: " << typeInfo.name << " v" << typeInfo.version;
        }

        const auto &boxes_dims = getInputShapeAtPort(NMS_BOXES).getDims();
        if (boxes_dims.size() != 3)
            IE_THROW() << errorPrefix << "has unsupported 'boxes' input rank: " << boxes_dims.size();
        if (boxes_dims[2] != 4)
            IE_THROW() << errorPrefix << "has unsupported 'boxes' input 3rd dimension size: " << boxes_dims[2];

        const auto &scores_dims = getInputShapeAtPort(NMS_SCORES).getDims();
        if (scores_dims.size() != 3)
            IE_THROW() << errorPrefix << "has unsupported 'scores' input rank: " << scores_dims.size();

        const Shape valid_outputs_shape = getOutputShapeAtPort(NMS_VALIDOUTPUTS);
        if (valid_outputs_shape.getRank() != 1)
            IE_THROW() << errorPrefix << "has unsupported 'valid_outputs' output rank: " << valid_outputs_shape.getRank();
        if (valid_outputs_shape.getDims()[0] != 1)
            IE_THROW() << errorPrefix << "has unsupported 'valid_outputs' output 1st dimension size: " << valid_outputs_shape.getDims()[1];
}

void MKLDNNNonMaxSuppressionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32, Precision::BF16};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_VALIDOUTPUTS), supportedIntOutputPrecision, "valid_outputs", outType);

    const std::vector<Precision> supportedPrecision = {Precision::I16, Precision::U8, Precision::I8, Precision::U16, Precision::I32,
                                                       Precision::U32, Precision::I64, Precision::U64};

    if (inputShapes.size() > NMS_MAXOUTPUTBOXESPERCLASS)
        check1DInput(getInputShapeAtPort(NMS_MAXOUTPUTBOXESPERCLASS), supportedPrecision, "max_output_boxes_per_class", NMS_MAXOUTPUTBOXESPERCLASS);
    if (inputShapes.size() > NMS_IOUTHRESHOLD)
        check1DInput(getInputShapeAtPort(NMS_IOUTHRESHOLD), supportedFloatPrecision, "iou_threshold", NMS_IOUTHRESHOLD);
    if (inputShapes.size() > NMS_SCORETHRESHOLD)
        check1DInput(getInputShapeAtPort(NMS_SCORETHRESHOLD), supportedFloatPrecision, "score_threshold", NMS_SCORETHRESHOLD);
    if (inputShapes.size() > NMS_SOFTNMSSIGMA)
        check1DInput(getInputShapeAtPort(NMS_SCORETHRESHOLD), supportedFloatPrecision, "soft_nms_sigma", NMS_SCORETHRESHOLD);

    checkOutput(getOutputShapeAtPort(NMS_SELECTEDINDICES), supportedIntOutputPrecision, "selected_indices", NMS_SELECTEDINDICES);
    checkOutput(getOutputShapeAtPort(NMS_SELECTEDSCORES), supportedFloatPrecision, "selected_scores", NMS_SELECTEDSCORES);

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (int i = 0; i < inputShapes.size(); ++i) {
        Precision inPrecision = i == NMS_MAXOUTPUTBOXESPERCLASS ? Precision::I32 : Precision::FP32;
        inDataConf.emplace_back(LayoutType::ncsp, inPrecision);
    }

    std::vector<PortConfigurator> outDataConf;
    outDataConf.reserve(outputShapes.size());
    for (int i = 0; i < outputShapes.size(); ++i) {
        Precision outPrecision = i == NMS_SELECTEDSCORES ? Precision::FP32 : Precision::I32;
        outDataConf.emplace_back(LayoutType::ncsp, outPrecision);
    }

    addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
}

void MKLDNNNonMaxSuppressionNode::prepareParams() {
    const auto& boxes_dims = isDynamicNode() ? getParentEdgesAtPort(NMS_BOXES)[0]->getMemory().getStaticDims() :
                                               getInputShapeAtPort(NMS_BOXES).getStaticDims();
    const auto& scores_dims = isDynamicNode() ? getParentEdgesAtPort(NMS_SCORES)[0]->getMemory().getStaticDims() :
                                                getInputShapeAtPort(NMS_SCORES).getStaticDims();

    num_batches = boxes_dims[0];
    num_boxes = boxes_dims[1];
    num_classes = scores_dims[1];
    if (num_batches != scores_dims[0])
        IE_THROW() << errorPrefix << " num_batches is different in 'boxes' and 'scores' inputs";
    if (num_boxes != scores_dims[2])
        IE_THROW() << errorPrefix << " num_boxes is different in 'boxes' and 'scores' inputs";

    numFiltBox.resize(num_batches);
    for (auto & i : numFiltBox)
        i.resize(num_classes);
}

bool MKLDNNNonMaxSuppressionNode::isExecutable() const {
    return isDynamicNode() || MKLDNNNode::isExecutable();
}

void MKLDNNNonMaxSuppressionNode::executeDynamicImpl(mkldnn::stream strm) {
    if (hasEmptyInputTensors() || (inputShapes.size() > NMS_MAXOUTPUTBOXESPERCLASS &&
            reinterpret_cast<int *>(getParentEdgeAt(NMS_MAXOUTPUTBOXESPERCLASS)->getMemoryPtr()->GetPtr())[0] == 0)) {
        getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getMemoryPtr()->redefineDesc(
            getBaseMemDescAtOutputPort(NMS_SELECTEDINDICES)->cloneWithNewDims({0, 3}));
        getChildEdgesAtPort(NMS_SELECTEDSCORES)[0]->getMemoryPtr()->redefineDesc(
            getBaseMemDescAtOutputPort(NMS_SELECTEDSCORES)->cloneWithNewDims({0, 3}));
        *reinterpret_cast<int *>(getChildEdgesAtPort(NMS_VALIDOUTPUTS)[0]->getMemoryPtr()->GetPtr()) = 0;
        return;
    }
    execute(strm);
}

void MKLDNNNonMaxSuppressionNode::execute(mkldnn::stream strm) {
    const float *boxes = reinterpret_cast<const float *>(getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->GetPtr());
    const float *scores = reinterpret_cast<const float *>(getParentEdgeAt(NMS_SCORES)->getMemoryPtr()->GetPtr());

    if (inputShapes.size() > NMS_MAXOUTPUTBOXESPERCLASS) {
        max_output_boxes_per_class = reinterpret_cast<int *>(getParentEdgeAt(NMS_MAXOUTPUTBOXESPERCLASS)->getMemoryPtr()->GetPtr())[0];
    }

    max_output_boxes_per_class = std::min(max_output_boxes_per_class, num_boxes);

    if (max_output_boxes_per_class == 0) {
        return;
    }

    if (inputShapes.size() > NMS_IOUTHRESHOLD)
        iou_threshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_IOUTHRESHOLD)->getMemoryPtr()->GetPtr())[0];

    if (inputShapes.size() > NMS_SCORETHRESHOLD)
        score_threshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_SCORETHRESHOLD)->getMemoryPtr()->GetPtr())[0];

    if (inputShapes.size() > NMS_SOFTNMSSIGMA)
        soft_nms_sigma = reinterpret_cast<float *>(getParentEdgeAt(NMS_SOFTNMSSIGMA)->getMemoryPtr()->GetPtr())[0];
    scale = 0.0f;
    if (soft_nms_sigma > 0.0) {
        scale = -0.5f / soft_nms_sigma;
    }

    auto boxesStrides = getParentEdgeAt(NMS_BOXES)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    auto scoresStrides = getParentEdgeAt(NMS_SCORES)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();

    const auto maxNumberOfBoxes = max_output_boxes_per_class * num_batches * num_classes;
    std::vector<filteredBoxes> filtBoxes(maxNumberOfBoxes);

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

    auto indicesMemPtr = getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getMemoryPtr();
    auto scoresMemPtr =  getChildEdgesAtPort(NMS_SELECTEDSCORES)[0]->getMemoryPtr();
    const size_t validOutputs = std::min(filtBoxes.size(), maxNumberOfBoxes);

    // TODO [DS NMS]: remove when nodes from models where nms is not last node in model supports DS
    if (isDynamicNode()) {
        VectorDims newDims{validOutputs, 3};
        indicesMemPtr->redefineDesc(getBaseMemDescAtOutputPort(NMS_SELECTEDINDICES)->cloneWithNewDims(newDims));
        scoresMemPtr->redefineDesc(getBaseMemDescAtOutputPort(NMS_SELECTEDSCORES)->cloneWithNewDims(newDims));
    }

    int selectedIndicesStride = indicesMemPtr->GetDescWithType<BlockedMemoryDesc>()->getStrides()[0];

    int *selectedIndicesPtr = reinterpret_cast<int *>(indicesMemPtr->GetPtr());
    float *selectedScoresPtr = reinterpret_cast<float *>(scoresMemPtr->GetPtr());

    size_t idx = 0lu;
    for (; idx < validOutputs; idx++) {
        selectedIndicesPtr[0] = filtBoxes[idx].batch_index;
        selectedIndicesPtr[1] = filtBoxes[idx].class_index;
        selectedIndicesPtr[2] = filtBoxes[idx].box_index;
        selectedIndicesPtr += selectedIndicesStride;

        selectedScoresPtr[0] = static_cast<float>(filtBoxes[idx].batch_index);
        selectedScoresPtr[1] = static_cast<float>(filtBoxes[idx].class_index);
        selectedScoresPtr[2] = static_cast<float>(filtBoxes[idx].score);
        selectedScoresPtr += selectedIndicesStride;
    }

    // TODO [DS NMS]: remove when nodes from models where nms is not last node in model supports DS
    if (!isDynamicNode()) {
        std::fill(selectedIndicesPtr, selectedIndicesPtr + (maxNumberOfBoxes - idx) * selectedIndicesStride, -1);
        std::fill(selectedScoresPtr, selectedScoresPtr + (maxNumberOfBoxes - idx) * selectedIndicesStride, -1.f);
    }

    int *valid_outputs = reinterpret_cast<int *>(getChildEdgesAtPort(NMS_VALIDOUTPUTS)[0]->getMemoryPtr()->GetPtr());
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

void MKLDNNNonMaxSuppressionNode::nmsWithSoftSigma(const float *boxes, const float *scores, const VectorDims &boxesStrides,
                                                             const VectorDims &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
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

void MKLDNNNonMaxSuppressionNode::nmsWithoutSoftSigma(const float *boxes, const float *scores, const VectorDims &boxesStrides,
                                                                const VectorDims &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
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

void MKLDNNNonMaxSuppressionNode::checkPrecision(const Precision& prec, const std::vector<Precision>& precList,
                                                           const std::string& name, const std::string& type) {
    if (std::find(precList.begin(), precList.end(), prec) == precList.end())
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' " << type << " precision: " << prec;
}

void MKLDNNNonMaxSuppressionNode::check1DInput(const Shape& shape, const std::vector<Precision>& precList,
                                                         const std::string& name, const size_t port) {
    checkPrecision(getOriginalInputPrecisionAtPort(port), precList, name, inType);

    if (shape.getRank() != 0 && shape.getRank() != 1)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' input rank: " << shape.getRank();
    if (shape.getRank() == 1)
        if (shape.getDims()[0] != 1)
            IE_THROW() << errorPrefix << "has unsupported '" << name << "' input 1st dimension size: " << MemoryDescUtils::dim2str(shape.getDims()[0]);
}

void MKLDNNNonMaxSuppressionNode::checkOutput(const Shape& shape, const std::vector<Precision>& precList,
                                                        const std::string& name, const size_t port) {
    checkPrecision(getOriginalOutputPrecisionAtPort(port), precList, name, outType);

    if (shape.getRank() != 2)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' output rank: " << shape.getRank();
    if (shape.getDims()[1] != 3)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' output 2nd dimension size: " << MemoryDescUtils::dim2str(shape.getDims()[1]);
}


REG_MKLDNN_PRIM_FOR(MKLDNNNonMaxSuppressionNode, NonMaxSuppression)
