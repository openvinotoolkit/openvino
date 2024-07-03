// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "embedding_bag_offsets.h"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "openvino/op/embeddingbag_offsets.hpp"


namespace ov {
namespace intel_cpu {
namespace node {

bool EmbeddingBagOffset::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto embBagOffsetSumOp = ov::as_type_ptr<const ov::op::v3::EmbeddingBagOffsetsSum>(op);
        const auto embBagOffsetOp = ov::as_type_ptr<const ov::op::v15::EmbeddingBagOffsets>(op);
        if (!embBagOffsetSumOp && !embBagOffsetOp) {
            errorMessage = "Node is not an instance of the v3::EmbeddingBagOffsetsSum or v15::EmbeddingBagOffsets operation.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

EmbeddingBagOffset::EmbeddingBagOffset(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)),
      EmbeddingBag(op, 3lu, 1lu, 4lu, 3lu) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    auto offsets_op = ov::as_type_ptr<ov::op::util::EmbeddingBagOffsetsBase>(op);
    if (offsets_op) {
        using OpReduction = ov::op::util::EmbeddingBagOffsetsBase::Reduction;
        switch (offsets_op->get_reduction()) {
        case OpReduction::SUM:
            _reduction = Reduction::SUM;
            break;
        case OpReduction::MEAN:
            _reduction = Reduction::MEAN;
            break;
        default:
            THROW_CPU_NODE_ERR("EmbeddingBagOffsets does not support reduction mode: ", ov::as_string(offsets_op->get_reduction()));
        }
    }
    if (getInputShapeAtPort(INDICES_IDX).getRank() != 1ul)
        OPENVINO_THROW("'", _layerName, "' layer has indices data with invalid rank.");

    if (getInputShapeAtPort(OFFSETS_IDX).getRank() != 1ul)
        OPENVINO_THROW("'", _layerName, "' layer's offsets data has invalid rank.");
}

void EmbeddingBagOffset::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::string logPrefix = std::string("Layer EmbeddingBag with name '") + _layerName + "' ";
    static const std::set<ov::element::Type > supportedPrecisions =
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
                                                       {LayoutType::ncsp, ov::element::i32}});
    if (inputShapes.size() > DEFAULT_INDEX_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, ov::element::i32});
    if (inputShapes.size() > PER_SAMPLE_WEIGHTS_IDX)
        inDataConfigurators.push_back({LayoutType::ncsp, inDataPrecision});

    addSupportedPrimDesc(inDataConfigurators, {{LayoutType::ncsp, inDataPrecision}}, impl_desc_type::ref_any);
}

void EmbeddingBagOffset::prepareParams() {
    _indicesLen = getParentEdgeAt(INDICES_IDX)->getMemory().getStaticDims()[0];
    _offsetsLen = getParentEdgeAt(OFFSETS_IDX)->getMemory().getStaticDims()[0];
    EmbeddingBag::prepareParams(getParentEdgeAt(EMB_TABLE_IDX)->getMemory().getStaticDims());
}

void EmbeddingBagOffset::initFromInputs() {
    indicesData_ = getSrcDataAtPortAs<const int>(INDICES_IDX);
    offsetsData_ = getSrcDataAtPortAs<const int>(OFFSETS_IDX);

    if (getParentEdges().size() > DEFAULT_INDEX_IDX && *getSrcDataAtPortAs<const int>(DEFAULT_INDEX_IDX) != -1) {
        defaultIndices_ = getSrcDataAtPortAs<const int>(DEFAULT_INDEX_IDX);
    }
}

void EmbeddingBagOffset::getIndices(size_t embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) {
    if (static_cast<size_t>(embIndex) >= _offsetsLen) {
        OPENVINO_THROW("Invalid embedding bag index.");
    }
    if (static_cast<size_t>(offsetsData_[embIndex]) >= _indicesLen) {
        OPENVINO_THROW("Offset value exceeds indices size.");
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

void EmbeddingBagOffset::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool EmbeddingBagOffset::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void EmbeddingBagOffset::execute(dnnl::stream strm) {
    const auto *srcData = getSrcDataAtPortAs<const uint8_t>(0);
    const uint8_t* weightsData = nullptr;
    if (_withWeights)
        weightsData = getSrcDataAtPortAs<const uint8_t>(PER_SAMPLE_WEIGHTS_IDX);

    const auto &inputMem  = getParentEdgeAt(0)->getMemory();
    EmbeddingBag::execute(srcData, weightsData, inputMem.getDesc().getPrecision(),
                                       inputMem.getStaticDims(), getDstMemoryAtPort(0));
}

bool EmbeddingBagOffset::created() const {
    return getType() == Type::EmbeddingBagOffsets;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
