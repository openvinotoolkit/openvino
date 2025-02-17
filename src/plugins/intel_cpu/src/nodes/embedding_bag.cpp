// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag.h"

#include <cmath>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "dnnl_types.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov::intel_cpu::node {

EmbeddingBag::EmbeddingBag(const std::shared_ptr<ov::Node>& op,
                           size_t requiredInputNum,
                           size_t indicesIdx,
                           size_t perSampleWeightsIdx,
                           size_t defaultIndexIdx)
    : INDICES_IDX(indicesIdx),
      PER_SAMPLE_WEIGHTS_IDX(perSampleWeightsIdx),
      DEFAULT_INDEX_IDX(defaultIndexIdx),
      _layerName(op->get_friendly_name()) {
    std::string logPrefix = std::string("Layer EmbeddingBag with name '") + _layerName + "' ";
    if (op->get_input_size() < requiredInputNum || op->get_output_size() != 1) {
        OPENVINO_THROW(logPrefix, "has incorrect number of input or output edges!");
    }
    if ((op->get_input_size() > PER_SAMPLE_WEIGHTS_IDX)) {
        _withWeights = true;
    }
    if (_withWeights) {
        if (op->get_input_shape(PER_SAMPLE_WEIGHTS_IDX) != op->get_input_shape(INDICES_IDX)) {
            OPENVINO_THROW(logPrefix, "must have equal shapes for indices and per_sample_weights inputs.");
        }
    }
}

void EmbeddingBag::prepareParams(const VectorDims& indexStaticShape) {
    _embDepth = 1lu;
    for (size_t i = 1lu; i < indexStaticShape.size(); i++) {
        _embDepth *= indexStaticShape[i];
    }
}

template <typename T>
void EmbeddingBag::processData(const T* srcData,
                               const T* weightsData,
                               const VectorDims& inDataDims,
                               const MemoryPtr& outMemory) {
    std::string msgPrefix = std::string("Node EmbeddingBag with name '") + _layerName + "' ";

    initFromInputs();

    const size_t outputBagsNum = outMemory->getShape().getStaticDims()[0];
    auto* dstData = outMemory->getDataAs<T>();

    auto threadBody = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(outputBagsNum, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }

        size_t indicesSize = 0lu;
        const int* indices = nullptr;
        int weightsIdx = 0lu;
        bool withWeights = _withWeights;

        for (size_t obi = start; obi < end; obi++) {
            size_t dstIndex = obi * _embDepth;
            getIndices(obi, indices, indicesSize, weightsIdx, withWeights);

            if (indices != nullptr) {
                withWeights = withWeights & _withWeights;

                size_t inIdx = 0lu;
                if (static_cast<size_t>(indices[inIdx]) >= inDataDims[0]) {
                    OPENVINO_THROW(msgPrefix + "' has invalid embedding bag index: " + std::to_string(indices[inIdx]));
                }
                size_t srcIndex = indices[inIdx] * _embDepth;

                if (withWeights) {
                    for (size_t i = 0lu; i < _embDepth; i++) {
                        dstData[dstIndex + i] = srcData[srcIndex + i] * weightsData[weightsIdx];
                    }
                    weightsIdx++;
                } else {
                    for (size_t i = 0lu; i < _embDepth; i++) {
                        dstData[dstIndex + i] = srcData[srcIndex + i];
                    }
                }

                for (inIdx = 1lu; inIdx < indicesSize; inIdx++) {
                    if (static_cast<size_t>(indices[inIdx]) >= inDataDims[0]) {
                        OPENVINO_THROW(msgPrefix +
                                       "' has invalid embedding bag index: " + std::to_string(indices[inIdx]));
                    }
                    size_t srcIndex = indices[inIdx] * _embDepth;

                    if (withWeights) {
                        for (size_t i = 0lu; i < _embDepth; i++) {
                            dstData[dstIndex + i] += srcData[srcIndex + i] * weightsData[weightsIdx];
                        }
                        weightsIdx++;
                    } else {
                        for (size_t i = 0lu; i < _embDepth; i++) {
                            dstData[dstIndex + i] += srcData[srcIndex + i];
                        }
                    }
                }
                if (_reduction == Reduction::MEAN) {
                    for (size_t i = 0lu; i < _embDepth; i++) {
                        dstData[dstIndex + i] /= indicesSize;
                    }
                }
            } else {
                for (size_t i = 0lu; i < _embDepth; i++) {
                    dstData[dstIndex + i] = 0;
                }
            }
        }
    };

    parallel_nt(0, threadBody);
}

void EmbeddingBag::execute(const uint8_t* srcData,
                           const uint8_t* weightsData,
                           const ov::element::Type& srcPrc,
                           const VectorDims& inDims,
                           const MemoryPtr& outMemory) {
    switch (srcPrc) {
    case ov::element::f32: {
        return processData<element_type_traits<ov::element::f32>::value_type>(
            reinterpret_cast<const float*>(srcData),
            reinterpret_cast<const float*>(weightsData),
            inDims,
            outMemory);
    }
    case ov::element::i8: {
        return processData<element_type_traits<ov::element::i8>::value_type>(
            reinterpret_cast<const int8_t*>(srcData),
            reinterpret_cast<const int8_t*>(weightsData),
            inDims,
            outMemory);
    }
    case ov::element::u8: {
        return processData<element_type_traits<ov::element::u8>::value_type>(srcData, weightsData, inDims, outMemory);
    }
    case ov::element::i32: {
        return processData<element_type_traits<ov::element::i32>::value_type>(
            reinterpret_cast<const int32_t*>(srcData),
            reinterpret_cast<const int32_t*>(weightsData),
            inDims,
            outMemory);
    }
    default: {
        OPENVINO_THROW("EmbeddingBag layer does not support precision '" + std::string(srcPrc.get_type_name()) + "'");
    }
    }
}

}  // namespace ov::intel_cpu::node
