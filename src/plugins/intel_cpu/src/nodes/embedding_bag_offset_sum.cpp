// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "embedding_bag_offset_sum.h"
#include <ngraph/opsets/opset3.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool EmbeddingBagOffsetSum::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto embBagOffsetSumOp = ngraph::as_type_ptr<const ngraph::op::v3::EmbeddingBagOffsetsSum>(op);
        if (!embBagOffsetSumOp) {
            errorMessage = "Node is not an instance of the EmbeddingBagOffsetsSum operation from opset v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

EmbeddingBagOffsetSum::EmbeddingBagOffsetSum(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)),
      EmbeddingBagSum(op, 3lu, 1lu, 4lu, 3lu) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (getInputShapeAtPort(INDICES_IDX).getRank() != 1ul)
        IE_THROW() << "'" << _layerName << "' layer has indices data with invalid rank.";

    if (getInputShapeAtPort(OFFSETS_IDX).getRank() != 1ul)
        IE_THROW() << "'" << _layerName << "' layer's offsets data has invalid rank.";
}

void EmbeddingBagOffsetSum::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::string logPrefix = std::string("Layer EmbeddingBagSum with name '") + _layerName + "' ";
    static const std::set<ov::element::Type > supportedPrecisions =
            {ov::element::f32, ov::element::i8, ov::element::u8, ov::element::i32};

    auto inDataPrecision = getOriginalInputPrecisionAtPort(EMB_TABLE_IDX);
    if (inDataPrecision == ov::element::bf16)
        inDataPrecision = ov::element::f32;
    if (!supportedPrecisions.empty()) {
        if (supportedPrecisions.find(inDataPrecision) == supportedPrecisions.end())
            IE_THROW() << logPrefix << "has unsupported precision: " << inDataPrecision.get_type_name();
    } else {
        static const std::set<ov::element::Type> defaultSupportedPrecisions =
                {ov::element::f32, ov::element::i8, ov::element::u8, ov::element::i32};
        if (defaultSupportedPrecisions.find(inDataPrecision) == defaultSupportedPrecisions.end())
            IE_THROW() << logPrefix << "has unsupported precision: " << inDataPrecision.get_type_name();
    }

    std::vector<PortConfigurator> inDataConfigurators({{LayoutType::ncsp, inDataPrecision},
                                                       {LayoutType::ncsp, ov::element::i32},
                                                       {LayoutType::ncsp, ov::element::i32}});
    if (inputShapes.size() > DEFAULT_INDEX_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, ov::element::i32});
    if (inputShapes.size() > PER_SAMPLE_WEIGHTS_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, inDataPrecision});

    addSupportedPrimDesc(inDataConfigurators, {{LayoutType::ncsp, inDataPrecision}}, impl_desc_type::ref_any);
}

void EmbeddingBagOffsetSum::prepareParams() {
    _indicesLen = getParentEdgesAtPort(INDICES_IDX)[0]->getMemory().getStaticDims()[0];
    _offsetsLen = getParentEdgesAtPort(OFFSETS_IDX)[0]->getMemory().getStaticDims()[0];
    EmbeddingBagSum::prepareParams(getParentEdgesAtPort(EMB_TABLE_IDX)[0]->getMemory().getStaticDims());
}

void EmbeddingBagOffsetSum::initFromInputs() {
    indicesData_ = reinterpret_cast<const int *>(getParentEdgeAt(INDICES_IDX)->getMemoryPtr()->getData());
    offsetsData_ = reinterpret_cast<const int *>(getParentEdgeAt(OFFSETS_IDX)->getMemoryPtr()->getData());

    if (getParentEdges().size() > DEFAULT_INDEX_IDX) {
        defaultIndices_ = reinterpret_cast<const int *>(getParentEdgeAt(DEFAULT_INDEX_IDX)->getMemoryPtr()->getData());
    }
}

void EmbeddingBagOffsetSum::getIndices(size_t embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) {
    if (static_cast<size_t>(embIndex) >= _offsetsLen) {
        IE_THROW() << "Invalid embedding bag index.";
    }
    if (static_cast<size_t>(offsetsData_[embIndex]) >= _indicesLen) {
        IE_THROW() << "Offset value exceeds indices size.";
    }

    indices = nullptr;
    size = 0lu;
    withWeight = _withWeights;

    if (embIndex == _offsetsLen - 1lu)
        size = _indicesLen - offsetsData_[embIndex];
    else
        size = offsetsData_[embIndex + 1lu] - offsetsData_[embIndex];

    if (size != 0lu) {
        indices = indicesData_ + offsetsData_[embIndex];
    } else {
        // Empty or default bag
        withWeight = false;
        if (defaultIndices_) {
            indices = defaultIndices_;
            size = 1lu;
        }
        return;
    }

    if (withWeight)
        weightsIdx = offsetsData_[embIndex];
}

void EmbeddingBagOffsetSum::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool EmbeddingBagOffsetSum::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void EmbeddingBagOffsetSum::execute(dnnl::stream strm) {
    const auto *srcData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(0)->getMemoryPtr()->getData());
    const uint8_t* weightsData = nullptr;
    if (_withWeights)
        weightsData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(PER_SAMPLE_WEIGHTS_IDX)->getMemoryPtr()->getData());

    const auto &inputMem  = getParentEdgeAt(0)->getMemory();
    EmbeddingBagSum::execute(srcData, weightsData, inputMem.getDesc().getPrecision(),
                                       inputMem.getStaticDims(), getChildEdgesAtPort(0)[0]->getMemoryPtr());
}

bool EmbeddingBagOffsetSum::created() const {
    return getType() == Type::EmbeddingBagOffsetsSum;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
