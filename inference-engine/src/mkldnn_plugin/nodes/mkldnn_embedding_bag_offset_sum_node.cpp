// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "mkldnn_embedding_bag_offset_sum_node.h"
#include <ngraph/opsets/opset3.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNEmbeddingBagOffsetSumNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto embBagOffsetSumOp = ngraph::as_type_ptr<const ngraph::op::v3::EmbeddingBagOffsetsSum>(op);
        if (!embBagOffsetSumOp) {
            errorMessage = "Node is not an instance of the EmbeddingBagOffsetsSum operation from opset v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNEmbeddingBagOffsetSumNode::MKLDNNEmbeddingBagOffsetSumNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache), MKLDNNEmbeddingBagSumNode(op, 3lu, 1lu, 4lu, 3lu) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (op->get_input_shape(INDICES_IDX).size() != 1)
        IE_THROW() << "'" << _layerName << "' layer has indices data with invalid shape.";

    if (op->get_input_shape(OFFSETS_IDX).size() != 1)
        IE_THROW() << "'" << _layerName << "' layer's offsets data has invalid shape.";

    _indicesLen = op->get_input_shape(INDICES_IDX)[0];
    _offsetsLen = op->get_input_shape(OFFSETS_IDX)[0];
}

void MKLDNNEmbeddingBagOffsetSumNode::initSupportedPrimitiveDescriptors() {
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

    std::vector<DataConfigurator> inDataConfigurators({{TensorDescCreatorTypes::ncsp, inDataPrecision},
                                                       {TensorDescCreatorTypes::ncsp, Precision::I32},
                                                       {TensorDescCreatorTypes::ncsp, Precision::I32}});
    if (getOriginalInputsNumber() > DEFAULT_INDEX_IDX)
        inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp, Precision::I32});
    if (getOriginalInputsNumber() > PER_SAMPLE_WEIGHTS_IDX)
        inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp, inDataPrecision});

    addSupportedPrimDesc(inDataConfigurators, {{TensorDescCreatorTypes::ncsp, inDataPrecision}}, impl_desc_type::ref_any);
}

void MKLDNNEmbeddingBagOffsetSumNode::initFromInputs() {
    indicesData_ = reinterpret_cast<const int *>(getParentEdgeAt(INDICES_IDX)->getMemoryPtr()->GetPtr());
    offsetsData_ = reinterpret_cast<const int *>(getParentEdgeAt(OFFSETS_IDX)->getMemoryPtr()->GetPtr());

    if (getParentEdges().size() > DEFAULT_INDEX_IDX) {
        defaultIndices_ = reinterpret_cast<const int *>(getParentEdgeAt(DEFAULT_INDEX_IDX)->getMemoryPtr()->GetPtr());
    }
}

void MKLDNNEmbeddingBagOffsetSumNode::getIndices(int embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) {
    if (embIndex >= _offsetsLen) {
        IE_THROW() << "Invalid embedding bag index.";
    }
    if (offsetsData_[embIndex] >= _indicesLen) {
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

void MKLDNNEmbeddingBagOffsetSumNode::execute(mkldnn::stream strm) {
    const auto *srcData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<uint8_t *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const uint8_t* weightsData = nullptr;
    if (_withWeights)
        weightsData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(PER_SAMPLE_WEIGHTS_IDX)->getMemoryPtr()->GetPtr());

    MKLDNNEmbeddingBagSumNode::execute(srcData, weightsData, dstData, getParentEdgeAt(0)->getDesc(), getChildEdgeAt(0)->getDesc());
}

bool MKLDNNEmbeddingBagOffsetSumNode::created() const {
    return getType() == EmbeddingBagOffsetsSum;
}

REG_MKLDNN_PRIM_FOR(MKLDNNEmbeddingBagOffsetSumNode, EmbeddingBagOffsetsSum)
