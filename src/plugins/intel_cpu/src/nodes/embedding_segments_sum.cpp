// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_segments_sum.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <set>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/embedding_bag.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/embedding_segments_sum.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool EmbeddingSegmentsSum::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                std::string& errorMessage) noexcept {
    try {
        const auto embBagSegSumOp = ov::as_type_ptr<const ov::op::v3::EmbeddingSegmentsSum>(op);
        if (!embBagSegSumOp) {
            errorMessage = "Node is not an instance of the v3 EmbeddingSegmentsSum operation";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

EmbeddingSegmentsSum::EmbeddingSegmentsSum(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)),
      EmbeddingBag(op, 4LU, 1LU, 5LU, 4LU) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    _reduction = Reduction::SUM;
    CPU_NODE_ASSERT(getInputShapeAtPort(INDICES_IDX).getRank() == 1UL,
                    "has indices data with invalid rank: ",
                    getInputShapeAtPort(INDICES_IDX).getRank());

    CPU_NODE_ASSERT(getInputShapeAtPort(SEGMENT_ID_IDX).getRank() == 1UL,
                    "has invalid segmentID data rank: ",
                    getInputShapeAtPort(SEGMENT_ID_IDX).getRank());
}

void EmbeddingSegmentsSum::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    static const std::set<ov::element::Type> supportedPrecisions = {ov::element::f32,
                                                                    ov::element::i8,
                                                                    ov::element::u8,
                                                                    ov::element::i32};

    auto inDataPrecision = getOriginalInputPrecisionAtPort(EMB_TABLE_IDX);
    if (any_of(inDataPrecision, ov::element::bf16, ov::element::f16)) {
        inDataPrecision = ov::element::f32;
    }
    if (!supportedPrecisions.empty()) {
        CPU_NODE_ASSERT(supportedPrecisions.find(inDataPrecision) != supportedPrecisions.end(),
                        "has unsupported precision: ",
                        inDataPrecision.get_type_name());
    } else {
        static const std::set<ov::element::Type> defaultSupportedPrecisions = {ov::element::f32,
                                                                               ov::element::i8,
                                                                               ov::element::u8,
                                                                               ov::element::i32};
        CPU_NODE_ASSERT(defaultSupportedPrecisions.find(inDataPrecision) != defaultSupportedPrecisions.end(),
                        "has unsupported precision: ",
                        inDataPrecision.get_type_name());
    }

    std::vector<PortConfigurator> inDataConfigurators({{LayoutType::ncsp, inDataPrecision},
                                                       {LayoutType::ncsp, ov::element::i32},
                                                       {LayoutType::ncsp, ov::element::i32},
                                                       {LayoutType::ncsp, ov::element::i32}});
    if (inputShapes.size() > DEFAULT_INDEX_IDX) {
        inDataConfigurators.emplace_back(LayoutType::ncsp, ov::element::i32);
    }
    if (inputShapes.size() > PER_SAMPLE_WEIGHTS_IDX) {
        inDataConfigurators.emplace_back(LayoutType::ncsp, inDataPrecision);
    }

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

void EmbeddingSegmentsSum::getIndices(size_t embIndex,
                                      const int*& indices,
                                      size_t& size,
                                      int& weightsIdx,
                                      bool& withWeight) {
    CPU_NODE_ASSERT(embIndex < static_cast<size_t>(lastNumSegments_), "Invalid embedding bag index.");

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
        size = 1LU;
        withWeight = false;
        if (defaultIndices_) {
            indices = defaultIndices_;
        }
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

    return lastNumSegments_ != getNumSegments();
}

void EmbeddingSegmentsSum::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool EmbeddingSegmentsSum::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0);
}

bool EmbeddingSegmentsSum::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void EmbeddingSegmentsSum::execute([[maybe_unused]] const dnnl::stream& strm) {
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

bool EmbeddingSegmentsSum::created() const {
    return getType() == Type::EmbeddingSegmentsSum;
}

}  // namespace ov::intel_cpu::node
