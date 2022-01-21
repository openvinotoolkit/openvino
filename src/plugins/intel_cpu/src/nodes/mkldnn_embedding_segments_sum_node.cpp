// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "mkldnn_embedding_segments_sum_node.h"
#include <ngraph/opsets/opset3.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNEmbeddingSegmentsSumNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto embBagSegSumOp = ngraph::as_type_ptr<const ngraph::op::v3::EmbeddingSegmentsSum>(op);
        if (!embBagSegSumOp) {
            errorMessage = "Node is not an instance of the EmbeddingSegmentsSum operation from opset v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNEmbeddingSegmentsSumNode::MKLDNNEmbeddingSegmentsSumNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache), MKLDNNEmbeddingBagSumNode(op, 4lu, 1lu, 5lu, 4lu) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    std::string errPrefix = std::string("EmbeddingSegmentsSum layer with name '") + _layerName + "' ";
    if (getInputShapeAtPort(INDICES_IDX).getRank() != 1ul)
        IE_THROW() << errPrefix << "has indices data with invalid rank: "
                   << getInputShapeAtPort(INDICES_IDX).getRank();

    if (getInputShapeAtPort(SEGMENT_ID_IDX).getRank() != 1ul)
        IE_THROW() << errPrefix << "has invalid segmentID data rank: "
                   << getInputShapeAtPort(SEGMENT_ID_IDX).getRank();
}

void MKLDNNEmbeddingSegmentsSumNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::string logPrefix = std::string("Layer EmbeddingBagSum with name '") + _layerName + "' ";
    static const std::set<Precision> supportedPrecisions =
            {Precision::FP32, Precision::I8, Precision::U8, Precision::I32};

    auto inDataPrecision = getOriginalInputPrecisionAtPort(EMB_TABLE_IDX);
    if (inDataPrecision == Precision::BF16)
        inDataPrecision = Precision::FP32;
    if (!supportedPrecisions.empty()) {
        if (supportedPrecisions.find(inDataPrecision) == supportedPrecisions.end())
            IE_THROW() << logPrefix << "has unsupported precision: " << inDataPrecision.name();
    } else {
        static const std::set<Precision> defaultSupportedPrecisions =
                {Precision::FP32, Precision::I8, Precision::U8, Precision::I32};
        if (defaultSupportedPrecisions.find(inDataPrecision) == defaultSupportedPrecisions.end())
            IE_THROW() << logPrefix << "has unsupported precision: " << inDataPrecision.name();
    }

    std::vector<PortConfigurator> inDataConfigurators({{LayoutType::ncsp, inDataPrecision},
                                                       {LayoutType::ncsp, Precision::I32},
                                                       {LayoutType::ncsp, Precision::I32},
                                                       {LayoutType::ncsp, Precision::I32}});
    if (inputShapes.size() > DEFAULT_INDEX_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, Precision::I32});
    if (inputShapes.size() > PER_SAMPLE_WEIGHTS_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, inDataPrecision});

    addSupportedPrimDesc(inDataConfigurators, {{LayoutType::ncsp, inDataPrecision}}, impl_desc_type::ref_any);
}

void MKLDNNEmbeddingSegmentsSumNode::prepareParams() {
    MKLDNNEmbeddingBagSumNode::prepareParams(getParentEdgesAtPort(EMB_TABLE_IDX)[0]->getMemory().getStaticDims());
}

void MKLDNNEmbeddingSegmentsSumNode::initFromInputs() {
    indices_ = reinterpret_cast<const int *>(getParentEdgeAt(INDICES_IDX)->getMemoryPtr()->GetPtr());
    indicesSize_ = getParentEdgeAt(INDICES_IDX)->getMemory().GetShape().getElementsCount();

    segmentIds_ = reinterpret_cast<const int *>(getParentEdgeAt(SEGMENT_ID_IDX)->getMemoryPtr()->GetPtr());

    if (getParentEdges().size() > NUM_SEGMENTS_IDX) {
        numSegments_ = reinterpret_cast<const int *>(getParentEdgeAt(NUM_SEGMENTS_IDX)->getMemoryPtr()->GetPtr())[0];
    }

    if (getParentEdges().size() > DEFAULT_INDEX_IDX) {
        defaultIndices_ = reinterpret_cast<const int *>(getParentEdgeAt(DEFAULT_INDEX_IDX)->getMemoryPtr()->GetPtr());
    }
}

void MKLDNNEmbeddingSegmentsSumNode::getIndices(int embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) {
    if (embIndex >= numSegments_)
        IE_THROW() << "Invalid embedding bag index.";

    indices = nullptr;
    size = 0;
    withWeight = true;

    for (int si = 0; si < indicesSize_; si++) {
        if (segmentIds_[si] == embIndex) {
            size++;
            if (indices == nullptr) {
                indices = indices_ + si;
                weightsIdx = si;
            }
        }
    }

    // Empty bag
    if (size == 0) {
        size = 1lu;
        withWeight = false;
        if (defaultIndices_)
            indices = defaultIndices_;
        return;
    }
}

std::vector<VectorDims> MKLDNNEmbeddingSegmentsSumNode::shapeInfer() const {
    return MKLDNNNode::shapeInferGeneric(PortMask(NUM_SEGMENTS_IDX));
}

void MKLDNNEmbeddingSegmentsSumNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNEmbeddingSegmentsSumNode::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void MKLDNNEmbeddingSegmentsSumNode::execute(mkldnn::stream strm) {
    const auto *srcData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<uint8_t *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const uint8_t* weightsData = nullptr;
    if (_withWeights)
        weightsData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(PER_SAMPLE_WEIGHTS_IDX)->getMemoryPtr()->GetPtr());

    const auto &inputMem  = getParentEdgeAt(0)->getMemory();
    MKLDNNEmbeddingBagSumNode::execute(srcData, weightsData, dstData, inputMem .getDesc().getPrecision(),
                                       inputMem .getStaticDims(), getChildEdgesAtPort(0)[0]->getMemory().GetShape().getStaticDims());
}

bool MKLDNNEmbeddingSegmentsSumNode::created() const {
    return getType() == EmbeddingSegmentsSum;
}

REG_MKLDNN_PRIM_FOR(MKLDNNEmbeddingSegmentsSumNode, EmbeddingSegmentsSum)
