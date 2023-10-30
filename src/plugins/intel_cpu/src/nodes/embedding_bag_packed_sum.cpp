// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "embedding_bag_packed_sum.h"
#include <ngraph/opsets/opset3.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool EmbeddingBagPackedSum::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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

EmbeddingBagPackedSum::EmbeddingBagPackedSum(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)),
      EmbeddingBagSum(op, 2lu, 1lu, 2lu, 3lu) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (getInputShapeAtPort(INDICES_IDX).getRank() != 2ul)
        OPENVINO_THROW("'", _layerName, "' layer has indices data with invalid rank.");
}

void EmbeddingBagPackedSum::initSupportedPrimitiveDescriptors() {
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
            OPENVINO_THROW(logPrefix, "has unsupported precision: ", inDataPrecision.name());
    } else {
        static const std::set<Precision> defaultSupportedPrecisions =
                {Precision::FP32, Precision::I8, Precision::U8, Precision::I32};
        if (defaultSupportedPrecisions.find(inDataPrecision) == defaultSupportedPrecisions.end())
            OPENVINO_THROW(logPrefix, "has unsupported precision: ", inDataPrecision.name());
    }

    std::vector<PortConfigurator> inDataConfigurators({{LayoutType::ncsp, inDataPrecision},
                                                       {LayoutType::ncsp, Precision::I32}});
    if (inputShapes.size() > PER_SAMPLE_WEIGHTS_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, inDataPrecision});

    addSupportedPrimDesc(inDataConfigurators, {{LayoutType::ncsp, inDataPrecision}}, impl_desc_type::ref_any);
}

void EmbeddingBagPackedSum::prepareParams() {
    _batch = getParentEdgesAtPort(INDICES_IDX)[0]->getMemory().getStaticDims()[0];
    _indicesPerBag = getParentEdgesAtPort(INDICES_IDX)[0]->getMemory().getStaticDims()[1];
    EmbeddingBagSum::prepareParams(getParentEdgesAtPort(EMB_TABLE_IDX)[0]->getMemory().getStaticDims());
}

void EmbeddingBagPackedSum::initFromInputs() {
    _indices = reinterpret_cast<const int *>(getParentEdgeAt(INDICES_IDX)->getMemoryPtr()->getData());
}

void EmbeddingBagPackedSum::getIndices(size_t embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) {
    if (static_cast<size_t>(embIndex) >= _batch * _indicesPerBag)
        OPENVINO_THROW("Invalid embedding bag index.");

    withWeight = true;

    indices = _indices + embIndex * _indicesPerBag;
    size = _indicesPerBag;

    weightsIdx = embIndex * _indicesPerBag;
}

void EmbeddingBagPackedSum::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool EmbeddingBagPackedSum::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void EmbeddingBagPackedSum::execute(dnnl::stream strm) {
    const auto *srcData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(0)->getMemoryPtr()->getData());
    const uint8_t* weightsData = nullptr;
    if (_withWeights)
        weightsData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(PER_SAMPLE_WEIGHTS_IDX)->getMemoryPtr()->getData());

    const auto &inputMem  = getParentEdgeAt(0)->getMemory();
    EmbeddingBagSum::execute(srcData, weightsData, inputMem.getDesc().getPrecision(),
                                       inputMem.getStaticDims(), getChildEdgesAtPort(0)[0]->getMemoryPtr());
}

bool EmbeddingBagPackedSum::created() const {
    return getType() == Type::EmbeddingBagPackedSum;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
