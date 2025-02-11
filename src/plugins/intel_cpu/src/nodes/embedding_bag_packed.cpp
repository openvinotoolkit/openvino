// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_packed.h"

#include <cmath>
#include <string>
#include <vector>

#include "openvino/op/embeddingbag_packed.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"

namespace ov::intel_cpu::node {

bool EmbeddingBagPacked::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                              std::string& errorMessage) noexcept {
    try {
        const auto embBagPackedSumOp = ov::as_type_ptr<const ov::op::v3::EmbeddingBagPackedSum>(op);
        const auto embBagPackedOp = ov::as_type_ptr<const ov::op::v15::EmbeddingBagPacked>(op);
        if (!embBagPackedSumOp && !embBagPackedOp) {
            errorMessage =
                "Node is not an instance of the v3::EmbeddingBagPackedSum or v15::EmbeddingBagPacked operations.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

EmbeddingBagPacked::EmbeddingBagPacked(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)),
      EmbeddingBag(op, 2lu, 1lu, 2lu, 3lu) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    auto packed_op = ov::as_type_ptr<ov::op::util::EmbeddingBagPackedBase>(op);
    if (packed_op) {
        using OpReduction = ov::op::util::EmbeddingBagPackedBase::Reduction;
        switch (packed_op->get_reduction()) {
        case OpReduction::SUM:
            _reduction = Reduction::SUM;
            break;
        case OpReduction::MEAN:
            _reduction = Reduction::MEAN;
            break;
        default:
            THROW_CPU_NODE_ERR("EmbeddingBagPacked does not support reduction mode: ",
                               ov::as_string(packed_op->get_reduction()));
        }
    }
    if (getInputShapeAtPort(INDICES_IDX).getRank() != 2ul) {
        THROW_CPU_NODE_ERR("has indices data with invalid rank.");
    }
}

void EmbeddingBagPacked::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    static const std::set<ov::element::Type> supportedPrecisions = {ov::element::f32,
                                                                    ov::element::i8,
                                                                    ov::element::u8,
                                                                    ov::element::i32};

    auto inDataPrecision = getOriginalInputPrecisionAtPort(EMB_TABLE_IDX);
    if (one_of(inDataPrecision, ov::element::bf16, ov::element::f16)) {
        inDataPrecision = ov::element::f32;
    }
    if (!supportedPrecisions.empty()) {
        if (supportedPrecisions.find(inDataPrecision) == supportedPrecisions.end()) {
            THROW_CPU_NODE_ERR("has unsupported precision: ", inDataPrecision.get_type_name());
        }
    } else {
        static const std::set<ov::element::Type> defaultSupportedPrecisions = {ov::element::f32,
                                                                               ov::element::i8,
                                                                               ov::element::u8,
                                                                               ov::element::i32};
        if (defaultSupportedPrecisions.find(inDataPrecision) == defaultSupportedPrecisions.end()) {
            THROW_CPU_NODE_ERR("has unsupported precision: ", inDataPrecision.get_type_name());
        }
    }

    std::vector<PortConfigurator> inDataConfigurators(
        {{LayoutType::ncsp, inDataPrecision}, {LayoutType::ncsp, ov::element::i32}});
    if (inputShapes.size() > PER_SAMPLE_WEIGHTS_IDX) {
        inDataConfigurators.emplace_back(LayoutType::ncsp, inDataPrecision);
    }

    addSupportedPrimDesc(inDataConfigurators, {{LayoutType::ncsp, inDataPrecision}}, impl_desc_type::ref_any);
}

void EmbeddingBagPacked::prepareParams() {
    _batch = getParentEdgeAt(INDICES_IDX)->getMemory().getStaticDims()[0];
    _indicesPerBag = getParentEdgeAt(INDICES_IDX)->getMemory().getStaticDims()[1];
    EmbeddingBag::prepareParams(getParentEdgeAt(EMB_TABLE_IDX)->getMemory().getStaticDims());
}

void EmbeddingBagPacked::initFromInputs() {
    _indices = getSrcDataAtPortAs<const int>(INDICES_IDX);
}

void EmbeddingBagPacked::getIndices(size_t embIndex,
                                    const int*& indices,
                                    size_t& size,
                                    int& weightsIdx,
                                    bool& withWeight) {
    if (static_cast<size_t>(embIndex) >= _batch * _indicesPerBag) {
        THROW_CPU_NODE_ERR("Invalid embedding bag index.");
    }

    withWeight = true;

    indices = _indices + embIndex * _indicesPerBag;
    size = _indicesPerBag;

    weightsIdx = embIndex * _indicesPerBag;
}

void EmbeddingBagPacked::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool EmbeddingBagPacked::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0);
}

bool EmbeddingBagPacked::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void EmbeddingBagPacked::execute(const dnnl::stream& strm) {
    const auto* srcData = getSrcDataAtPortAs<const uint8_t>(0);
    const uint8_t* weightsData = nullptr;
    if (_withWeights) {
        weightsData = getSrcDataAtPortAs<const uint8_t>(PER_SAMPLE_WEIGHTS_IDX);
    }

    const auto& inputMem = getParentEdgeAt(0)->getMemory();
    EmbeddingBag::execute(srcData,
                          weightsData,
                          inputMem.getDesc().getPrecision(),
                          inputMem.getStaticDims(),
                          getDstMemoryAtPort(0));
}

bool EmbeddingBagPacked::created() const {
    return getType() == Type::EmbeddingBagPacked;
}

}  // namespace ov::intel_cpu::node
