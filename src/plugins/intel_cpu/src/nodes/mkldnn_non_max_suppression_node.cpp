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

#include "cpu/x64/jit_generator.hpp"

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
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache), isSoftSuppressedByIOU(true) {
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
            boxEncodingType = static_cast<NMSBoxEncodeType>(nms5->get_box_encoding());
            sortResultDescending = nms5->get_sort_result_descending();
        // TODO [DS NMS]: remove when nodes from models where nms is not last node in model supports DS
        } else if (const auto nmsIe = std::dynamic_pointer_cast<const ngraph::op::internal::NonMaxSuppressionIEInternal>(op)) {
            boxEncodingType = nmsIe->m_center_point_box ? NMSBoxEncodeType::CENTER : NMSBoxEncodeType::CORNER;
            sortResultDescending = nmsIe->m_sort_result_descending;
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

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    addSupportedPrimDesc(inDataConf, outDataConf, impl_type);

    // as only FP32 and ncsp is supported, and kernel is shape agnostic, we can create here. There is no need to recompilation.
    createJitKernel();
}

void MKLDNNNonMaxSuppressionNode::prepareParams() {
    const auto& boxesDims = isDynamicNode() ? getParentEdgesAtPort(NMS_BOXES)[0]->getMemory().getStaticDims() :
                                               getInputShapeAtPort(NMS_BOXES).getStaticDims();
    const auto& scoresDims = isDynamicNode() ? getParentEdgesAtPort(NMS_SCORES)[0]->getMemory().getStaticDims() :
                                                getInputShapeAtPort(NMS_SCORES).getStaticDims();

    numBatches = boxesDims[0];
    numBoxes = boxesDims[1];
    numClasses = scoresDims[1];
    if (numBatches != scoresDims[0])
        IE_THROW() << errorPrefix << " numBatches is different in 'boxes' and 'scores' inputs";
    if (numBoxes != scoresDims[2])
        IE_THROW() << errorPrefix << " numBoxes is different in 'boxes' and 'scores' inputs";

    numFiltBox.resize(numBatches);
    for (auto & i : numFiltBox)
        i.resize(numClasses);
}

bool MKLDNNNonMaxSuppressionNode::isExecutable() const {
    return isDynamicNode() || MKLDNNNode::isExecutable();
}

void MKLDNNNonMaxSuppressionNode::createJitKernel() {
    auto jcp = jit_nms_config_params();
    jcp.box_encode_type = boxEncodingType;
    jcp.is_soft_suppressed_by_iou = isSoftSuppressedByIOU;

    if (mayiuse(cpu::x64::avx512_common)) {
        nms_kernel.reset(new jit_uni_nms_kernel_f32<cpu::x64::avx512_common>(jcp));
        eltsInVmm = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        nms_kernel.reset(new jit_uni_nms_kernel_f32<cpu::x64::avx2>(jcp));
        eltsInVmm = 8;
    } else if (mayiuse(cpu::x64::sse41)) {
        nms_kernel.reset(new jit_uni_nms_kernel_f32<cpu::x64::sse41>(jcp));
        eltsInVmm = 4;
    }

    if (nms_kernel)
        nms_kernel->create_ker();
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
        maxOutputBoxesPerClass = reinterpret_cast<int *>(getParentEdgeAt(NMS_MAXOUTPUTBOXESPERCLASS)->getMemoryPtr()->GetPtr())[0];
    }

    maxOutputBoxesPerClass = std::min(maxOutputBoxesPerClass, numBoxes);

    if (maxOutputBoxesPerClass == 0) {
        return;
    }

    if (inputShapes.size() > NMS_IOUTHRESHOLD)
        iouThreshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_IOUTHRESHOLD)->getMemoryPtr()->GetPtr())[0];

    if (inputShapes.size() > NMS_SCORETHRESHOLD)
        scoreThreshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_SCORETHRESHOLD)->getMemoryPtr()->GetPtr())[0];

    if (inputShapes.size() > NMS_SOFTNMSSIGMA)
        softNMSSigma = reinterpret_cast<float *>(getParentEdgeAt(NMS_SOFTNMSSIGMA)->getMemoryPtr()->GetPtr())[0];
    scale = 0.0f;
    if (softNMSSigma > 0.0) {
        scale = -0.5f / softNMSSigma;
    }

    auto boxesStrides = getParentEdgeAt(NMS_BOXES)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    auto scoresStrides = getParentEdgeAt(NMS_SCORES)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();

    const auto maxNumberOfBoxes = maxOutputBoxesPerClass * numBatches * numClasses;
    std::vector<filteredBoxes> filtBoxes(maxNumberOfBoxes);

    if (softNMSSigma == 0.0f) {
        nmsWithoutSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
    } else {
        nmsWithSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
    }

    size_t startOffset = numFiltBox[0][0];
    for (size_t b = 0; b < numFiltBox.size(); b++) {
        size_t batchOffset = b*numClasses*maxOutputBoxesPerClass;
        for (size_t c = (b == 0 ? 1 : 0); c < numFiltBox[b].size(); c++) {
            size_t offset = batchOffset + c*maxOutputBoxesPerClass;
            for (size_t i = 0; i < numFiltBox[b][c]; i++) {
                filtBoxes[startOffset + i] = filtBoxes[offset + i];
            }
            startOffset += numFiltBox[b][c];
        }
    }
    filtBoxes.resize(startOffset);

    // need more particular comparator to get deterministic behaviour
    // escape situation when filtred boxes with same score have different position from launch to launch
    if (sortResultDescending) {
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

    // update score, if iou is 0, weight is 1, score does not change
    // if is_soft_suppressed_by_iou is false, apply for all iou, including iou>iou_threshold, soft suppressed when score < score_threshold
    // if is_soft_suppressed_by_iou is true, hard suppressed by iou_threshold, then soft suppress
    auto coeff = [&](float iou) {
        if (isSoftSuppressedByIOU && iou > iouThreshold)
            return 0.0f;
        return std::exp(scale * iou * iou);
    };

    parallel_for2d(numBatches, numClasses, [&](int batch_idx, int class_idx) {
        std::vector<filteredBoxes> selectedBoxes;
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];
        std::vector<float> boxCoord0, boxCoord1, boxCoord2, boxCoord3;

        std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);  // score, box_id, suppress_begin_index
        for (int box_idx = 0; box_idx < numBoxes; box_idx++) {
            if (scoresPtr[box_idx] > scoreThreshold)
                sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
        }

        selectedBoxes.reserve(sorted_boxes.size());
        auto arg = jit_nms_args();
        arg.iou_threshold = static_cast<float*>(&iouThreshold);
        arg.score_threshold = static_cast<float*>(&scoreThreshold);
        arg.scale = static_cast<float*>(&scale);
        if (sorted_boxes.size() > 0) {
            // include first directly
            boxInfo candidateBox = sorted_boxes.top();
            sorted_boxes.pop();
            selectedBoxes.push_back({ candidateBox.score, batch_idx, class_idx, candidateBox.idx });
            if (nms_kernel) {
                boxCoord0.push_back(boxesPtr[candidateBox.idx * 4]);
                boxCoord1.push_back(boxesPtr[candidateBox.idx * 4 + 1]);
                boxCoord2.push_back(boxesPtr[candidateBox.idx * 4 + 2]);
                boxCoord3.push_back(boxesPtr[candidateBox.idx * 4 + 3]);
            }

            while (selectedBoxes.size() < maxOutputBoxesPerClass && !sorted_boxes.empty()) {
                boxInfo candidateBox = sorted_boxes.top();
                float origScore = candidateBox.score;
                sorted_boxes.pop();

                int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected, 2 for updated
                if (nms_kernel) {
                    arg.score = static_cast<float*>(&candidateBox.score);
                    arg.selected_boxes_num = selectedBoxes.size() - candidateBox.suppress_begin_index;
                    arg.selected_boxes_coord[0] = static_cast<float*>(&boxCoord0[candidateBox.suppress_begin_index]);
                    arg.selected_boxes_coord[1] = static_cast<float*>(&boxCoord1[candidateBox.suppress_begin_index]);
                    arg.selected_boxes_coord[2] = static_cast<float*>(&boxCoord2[candidateBox.suppress_begin_index]);
                    arg.selected_boxes_coord[3] = static_cast<float*>(&boxCoord3[candidateBox.suppress_begin_index]);
                    arg.candidate_box = static_cast<const float*>(&boxesPtr[candidateBox.idx * 4]);
                    arg.candidate_status = static_cast<int*>(&candidateStatus);
                    (*nms_kernel)(&arg);
                } else {
                    for (int selected_idx = static_cast<int>(selectedBoxes.size()) - 1; selected_idx >= candidateBox.suppress_begin_index; selected_idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[candidateBox.idx * 4], &boxesPtr[selectedBoxes[selected_idx].box_index * 4]);

                        // when is_soft_suppressed_by_iou is true, score is decayed to zero and implicitely suppressed if iou > iou_threshold.
                        candidateBox.score *= coeff(iou);
                        // soft suppressed
                        if (candidateBox.score <= scoreThreshold) {
                            candidateStatus = NMSCandidateStatus::SUPPRESSED;
                            break;
                        }
                    }
                }

                if (candidateStatus == NMSCandidateStatus::SUPPRESSED) {
                    continue;
                } else {
                    if (candidateBox.score == origScore) {
                        selectedBoxes.push_back({ candidateBox.score, batch_idx, class_idx, candidateBox.idx });
                        if (nms_kernel) {
                            boxCoord0.push_back(boxesPtr[candidateBox.idx * 4]);
                            boxCoord1.push_back(boxesPtr[candidateBox.idx * 4 + 1]);
                            boxCoord2.push_back(boxesPtr[candidateBox.idx * 4 + 2]);
                            boxCoord3.push_back(boxesPtr[candidateBox.idx * 4 + 3]);
                        }
                    } else {
                        candidateBox.suppress_begin_index = selectedBoxes.size();
                        sorted_boxes.push(candidateBox);
                    }
                }
            }
        }
        numFiltBox[batch_idx][class_idx] = selectedBoxes.size();
        size_t offset = batch_idx*numClasses*maxOutputBoxesPerClass + class_idx*maxOutputBoxesPerClass;
        for (size_t i = 0; i < selectedBoxes.size(); i++) {
            filtBoxes[offset + i] = selectedBoxes[i];
        }
    });
}

void MKLDNNNonMaxSuppressionNode::nmsWithoutSoftSigma(const float *boxes, const float *scores, const VectorDims &boxesStrides,
                                                                const VectorDims &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
    int max_out_box = static_cast<int>(maxOutputBoxesPerClass);
    parallel_for2d(numBatches, numClasses, [&](int batch_idx, int class_idx) {
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::vector<std::pair<float, int>> sorted_boxes;  // score, box_idx
        for (int box_idx = 0; box_idx < numBoxes; box_idx++) {
            if (scoresPtr[box_idx] > scoreThreshold)
                sorted_boxes.emplace_back(std::make_pair(scoresPtr[box_idx], box_idx));
        }

        int io_selection_size = 0;
        size_t sortedBoxSize = sorted_boxes.size();
        if (sortedBoxSize > 0) {
            parallel_sort(sorted_boxes.begin(), sorted_boxes.end(),
                          [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                              return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                          });
            int offset = batch_idx*numClasses*maxOutputBoxesPerClass + class_idx*maxOutputBoxesPerClass;
            filtBoxes[offset + 0] = filteredBoxes(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
            if (nms_kernel) {
                std::vector<float> boxCoord0(sortedBoxSize, 0.0f);
                std::vector<float> boxCoord1(sortedBoxSize, 0.0f);
                std::vector<float> boxCoord2(sortedBoxSize, 0.0f);
                std::vector<float> boxCoord3(sortedBoxSize, 0.0f);

                boxCoord0[io_selection_size] = boxesPtr[sorted_boxes[0].second * 4];
                boxCoord1[io_selection_size] = boxesPtr[sorted_boxes[0].second * 4 + 1];
                boxCoord2[io_selection_size] = boxesPtr[sorted_boxes[0].second * 4 + 2];
                boxCoord3[io_selection_size] = boxesPtr[sorted_boxes[0].second * 4 + 3];
                io_selection_size++;

                auto arg = jit_nms_args();
                arg.iou_threshold = static_cast<float*>(&iouThreshold);
                arg.score_threshold = static_cast<float*>(&scoreThreshold);
                arg.scale = static_cast<float*>(&scale);
                // box start index do not change for hard supresion
                arg.selected_boxes_coord[0] = static_cast<float*>(&boxCoord0[0]);
                arg.selected_boxes_coord[1] = static_cast<float*>(&boxCoord1[0]);
                arg.selected_boxes_coord[2] = static_cast<float*>(&boxCoord2[0]);
                arg.selected_boxes_coord[3] = static_cast<float*>(&boxCoord3[0]);

                for (size_t candidate_idx = 1; (candidate_idx < sortedBoxSize) && (io_selection_size < max_out_box); candidate_idx++) {
                    int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected
                    if (io_selection_size >= eltsInVmm) {
                        arg.selected_boxes_num = io_selection_size;
                        arg.candidate_box = static_cast<const float*>(&boxesPtr[sorted_boxes[candidate_idx].second * 4]);
                        arg.candidate_status = static_cast<int*>(&candidateStatus);
                        (*nms_kernel)(&arg);
                    } else {
                        for (int selected_idx = io_selection_size - 1; selected_idx >= 0; selected_idx--) {
                            float iou = intersectionOverUnion(&boxesPtr[sorted_boxes[candidate_idx].second * 4],
                                &boxesPtr[filtBoxes[offset + selected_idx].box_index * 4]);
                            if (iou >= iouThreshold) {
                                candidateStatus = NMSCandidateStatus::SUPPRESSED;
                                break;
                            }
                        }
                    }
                    if (candidateStatus == NMSCandidateStatus::SELECTED) {
                        boxCoord0[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * 4];
                        boxCoord1[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * 4 + 1];
                        boxCoord2[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * 4 + 2];
                        boxCoord3[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * 4 + 3];
                        filtBoxes[offset + io_selection_size] =
                            filteredBoxes(sorted_boxes[candidate_idx].first, batch_idx, class_idx, sorted_boxes[candidate_idx].second);
                        io_selection_size++;
                    }
                }
            } else {
                for (size_t candidate_idx = 1; (candidate_idx < sortedBoxSize) && (io_selection_size < max_out_box); candidate_idx++) {
                    int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected
                    for (int selected_idx = io_selection_size - 1; selected_idx >= 0; selected_idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[sorted_boxes[candidate_idx].second * 4],
                            &boxesPtr[filtBoxes[offset + selected_idx].box_index * 4]);
                        if (iou >= iouThreshold) {
                            candidateStatus = NMSCandidateStatus::SUPPRESSED;
                            break;
                        }
                    }

                    if (candidateStatus == NMSCandidateStatus::SELECTED) {
                        filtBoxes[offset + io_selection_size] =
                            filteredBoxes(sorted_boxes[candidate_idx].first, batch_idx, class_idx, sorted_boxes[candidate_idx].second);
                        io_selection_size++;
                    }
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
