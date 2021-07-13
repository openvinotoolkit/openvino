// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "mkldnn_embedding_bag_packed_sum_node.h"
#include <ngraph/opsets/opset3.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNEmbeddingBagPackedSumNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto embBagPackedSumOp = ngraph::as_type_ptr<const ngraph::op::v3::EmbeddingBagPackedSum>(op);
        if (!embBagPackedSumOp) {
            errorMessage = "Node is not an instance of the EmbeddingBagPackedSum operation from opset v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNEmbeddingBagPackedSumNode::MKLDNNEmbeddingBagPackedSumNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache), MKLDNNEmbeddingBagSumNode(op, 2lu, 1lu, 2lu, 3lu) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (op->get_input_shape(INDICES_IDX).size() != 2)
        IE_THROW() << "'" << _layerName << "' layer has indices data with invalid shape.";
    _batch = op->get_input_shape(INDICES_IDX)[0];
    _indicesPerBag = op->get_input_shape(INDICES_IDX)[1];
}

void MKLDNNEmbeddingBagPackedSumNode::initSupportedPrimitiveDescriptors() {
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
                                                       {TensorDescCreatorTypes::ncsp, Precision::I32}});
    if (getOriginalInputsNumber() > PER_SAMPLE_WEIGHTS_IDX)
        inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp, inDataPrecision});

    addSupportedPrimDesc(inDataConfigurators, {{TensorDescCreatorTypes::ncsp, inDataPrecision}}, impl_desc_type::ref_any);
}

void MKLDNNEmbeddingBagPackedSumNode::initFromInputs() {
    _indices = reinterpret_cast<const int *>(getParentEdgeAt(INDICES_IDX)->getMemoryPtr()->GetPtr());
}

void MKLDNNEmbeddingBagPackedSumNode::getIndices(int embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) {
    if (embIndex >= _batch * _indicesPerBag)
        IE_THROW() << "Invalid embedding bag index.";

    withWeight = true;

    indices = _indices + embIndex * _indicesPerBag;
    size = _indicesPerBag;

    weightsIdx = embIndex * _indicesPerBag;
}

void MKLDNNEmbeddingBagPackedSumNode::execute(mkldnn::stream strm) {
    const auto *srcData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<uint8_t *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const uint8_t* weightsData = nullptr;
    if (_withWeights)
        weightsData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(PER_SAMPLE_WEIGHTS_IDX)->getMemoryPtr()->GetPtr());

    MKLDNNEmbeddingBagSumNode::execute(srcData, weightsData, dstData, getParentEdgeAt(0)->getDesc(), getChildEdgeAt(0)->getDesc());
}

bool MKLDNNEmbeddingBagPackedSumNode::created() const {
    return getType() == EmbeddingBagPackedSum;
}

REG_MKLDNN_PRIM_FOR(MKLDNNEmbeddingBagPackedSumNode, EmbeddingBagPackedSum)
