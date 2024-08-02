// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "embedding_segments_sum.h"
#include "openvino/opsets/opset3.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool EmbeddingSegmentsSum::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto embBagSegSumOp = ov::as_type_ptr<const ov::op::v3::EmbeddingSegmentsSum>(op);
        if (!embBagSegSumOp) {
            errorMessage = "Node is not an instance of the EmbeddingSegmentsSum operation from opset v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

EmbeddingSegmentsSum::EmbeddingSegmentsSum(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, PortMask(NUM_SEGMENTS_IDX))),
      EmbeddingBag(op, 4lu, 1lu, 5lu, 4lu) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    _reduction = Reduction::SUM;
    std::string errPrefix = std::string("EmbeddingSegmentsSum layer with name '") + _layerName + "' ";
    if (getInputShapeAtPort(INDICES_IDX).getRank() != 1ul)
        OPENVINO_THROW(errPrefix, "has indices data with invalid rank: ", getInputShapeAtPort(INDICES_IDX).getRank());

    if (getInputShapeAtPort(SEGMENT_ID_IDX).getRank() != 1ul)
        OPENVINO_THROW(errPrefix, "has invalid segmentID data rank: ", getInputShapeAtPort(SEGMENT_ID_IDX).getRank());
}

void EmbeddingSegmentsSum::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::string logPrefix = std::string("Layer EmbeddingBag with name '") + _layerName + "' ";
    static const std::set<ov::element::Type> supportedPrecisions =
            {ov::element::f32, ov::element::i8, ov::element::u8, ov::element::i32};

    auto inDataPrecision = getOriginalInputPrecisionAtPort(EMB_TABLE_IDX);
    if (one_of(inDataPrecision, ov::element::bf16, ov::element::f16))
        inDataPrecision = ov::element::f32;
    if (!supportedPrecisions.empty()) {
        if (supportedPrecisions.find(inDataPrecision) == supportedPrecisions.end())
            OPENVINO_THROW(logPrefix, "has unsupported precision: ", inDataPrecision.get_type_name());
    } else {
        static const std::set<ov::element::Type> defaultSupportedPrecisions =
                {ov::element::f32, ov::element::i8, ov::element::u8, ov::element::i32};
        if (defaultSupportedPrecisions.find(inDataPrecision) == defaultSupportedPrecisions.end())
            OPENVINO_THROW(logPrefix, "has unsupported precision: ", inDataPrecision.get_type_name());
    }

    std::vector<PortConfigurator> inDataConfigurators({{LayoutType::ncsp, inDataPrecision},
                                                       {LayoutType::ncsp, ov::element::i32},
                                                       {LayoutType::ncsp, ov::element::i32},
                                                       {LayoutType::ncsp, ov::element::i32}});
    if (inputShapes.size() > DEFAULT_INDEX_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, ov::element::i32});
    if (inputShapes.size() > PER_SAMPLE_WEIGHTS_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, inDataPrecision});

    addSupportedPrimDesc(inDataConfigurators, {{LayoutType::ncsp, inDataPrecision}}, impl_desc_type::ref_any);
}

void EmbeddingSegmentsSum::prepareParams() {
    EmbeddingBag::prepareParams(getParentEdgeAt(EMB_TABLE_IDX)->getMemory().getStaticDims());
}

void EmbeddingSegmentsSum::initFromInputs() {
    indices_ = getSrcDataAtPortAs<const int>(INDICES_IDX);
    indicesSize_ = getParentEdgeAt(INDICES_IDX)->getMemory().getShape().getElementsCount();

    segmentIds_ = getSrcDataAtPortAs<const int>(SEGMENT_ID_IDX);
    lastNumSegments_ = getNumSegments();

    if (getParentEdges().size() > DEFAULT_INDEX_IDX) {
        defaultIndices_ = getSrcDataAtPortAs<const int>(DEFAULT_INDEX_IDX);
    }
}

void EmbeddingSegmentsSum::getIndices(size_t embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) {
    if (embIndex >= static_cast<size_t>(lastNumSegments_))
        OPENVINO_THROW("Invalid embedding bag index.");

    indices = nullptr;
    size = 0;
    withWeight = true;

    for (int si = 0; si < static_cast<int>(indicesSize_); si++) {
        if (static_cast<size_t>(segmentIds_[si]) == embIndex) {
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

int32_t EmbeddingSegmentsSum::getNumSegments() const {
    return getSrcDataAtPortAs<const int32_t>(NUM_SEGMENTS_IDX)[0];
}

bool EmbeddingSegmentsSum::needShapeInfer() const {
    if (Node::inputShapesModified()) {
        return true;
    }

    if (lastNumSegments_ != getNumSegments()) {
        return true;
    }

    return false;
}

void EmbeddingSegmentsSum::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool EmbeddingSegmentsSum::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void EmbeddingSegmentsSum::execute(dnnl::stream strm) {
    const auto *srcData = getSrcDataAtPortAs<const uint8_t>(0);
    const uint8_t* weightsData = nullptr;
    if (_withWeights)
        weightsData = getSrcDataAtPortAs<const uint8_t>(PER_SAMPLE_WEIGHTS_IDX);

    const auto &inputMem  = getParentEdgeAt(0)->getMemory();
    EmbeddingBag::execute(srcData, weightsData, inputMem.getDesc().getPrecision(),
                                       inputMem.getStaticDims(), getDstMemoryAtPort(0));
}

bool EmbeddingSegmentsSum::created() const {
    return getType() == Type::EmbeddingSegmentsSum;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
