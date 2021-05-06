// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_embedding_bag_sum_node.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNEmbeddingBagSumNode::MKLDNNEmbeddingBagSumNode(
            const std::shared_ptr<ngraph::Node>& op,
            size_t requiredInputNum,
            size_t indicesIdx,
            size_t perSampleWeightsIdx,
            size_t defaultIndexIdx) :
                INDICES_IDX(indicesIdx),
                PER_SAMPLE_WEIGHTS_IDX(perSampleWeightsIdx),
                DEFAULT_INDEX_IDX(defaultIndexIdx) {
    _layerName = op->get_friendly_name();
    std::string logPrefix = std::string("Layer EmbeddingBagSum with name '") + _layerName + "' ";
    if (op->get_input_size() < requiredInputNum || op->get_output_size() != 1)
        IE_THROW() << logPrefix << "has incorrect number of input or output edges!";

    if (op->get_input_size() > PER_SAMPLE_WEIGHTS_IDX)
        _withWeights = true;
    if (_withWeights) {
        if (op->get_input_shape(PER_SAMPLE_WEIGHTS_IDX) != op->get_input_shape(INDICES_IDX))
             IE_THROW() << logPrefix << "must have equal shapes for indices and per_sample_weights inputs.";
    }

    const auto& inDataDims = op->get_input_shape(EMB_TABLE_IDX);
    _embDepth = 1lu;
    for (size_t i = 1lu; i < inDataDims.size(); i++) {
        _embDepth *= inDataDims[i];
    }
}

template<typename T>
void MKLDNNEmbeddingBagSumNode::processData(const T* srcData, const T* weightsData, T* dstData,
                                            const InferenceEngine::TensorDesc& srcDesc, const InferenceEngine::TensorDesc& dstDesc) {
    std::string msgPrefix = std::string("Node EmbeddingBagSum with name '") + _layerName + "' ";

    initFromInputs();

    const auto& inDataDims = srcDesc.getDims();
    const size_t outputBagsNum = dstDesc.getDims()[0];

    auto threadBody = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(outputBagsNum, nthr, ithr, start, end);
        if (start >= end)
            return;

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
                if (indices[inIdx] >= inDataDims[0]) {
                    IE_THROW() << msgPrefix + "' has invalid embedding bag index: " + std::to_string(indices[inIdx]);
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
                    if (indices[inIdx] >= inDataDims[0]) {
                        IE_THROW() << msgPrefix + "' has invalid embedding bag index: " + std::to_string(indices[inIdx]);
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
            } else {
                for (size_t i = 0lu; i < _embDepth; i++) {
                    dstData[dstIndex + i] = 0;
                }
            }
        }
    };

    parallel_nt(0, threadBody);
}

void MKLDNNEmbeddingBagSumNode::execute(const uint8_t* srcData, const uint8_t* weightsData, uint8_t* dstData,
                                        const InferenceEngine::TensorDesc& srcDesc, const InferenceEngine::TensorDesc& dstDesc) {
    switch (srcDesc.getPrecision()) {
        case Precision::FP32: {
            return processData<PrecisionTrait<Precision::FP32>::value_type>(reinterpret_cast<const float*>(srcData),
                    reinterpret_cast<const float*>(weightsData), reinterpret_cast<float*>(dstData), srcDesc, dstDesc);
        }
        case Precision::I8: {
            return processData<PrecisionTrait<Precision::I8>::value_type>(reinterpret_cast<const int8_t*>(srcData),
                    reinterpret_cast<const int8_t*>(weightsData), reinterpret_cast<int8_t*>(dstData), srcDesc, dstDesc);
        }
        case Precision::U8: {
            return processData<PrecisionTrait<Precision::U8>::value_type>(srcData, weightsData, dstData, srcDesc, dstDesc);
        }
        case Precision::I32: {
            return processData<PrecisionTrait<Precision::I32>::value_type>(reinterpret_cast<const int32_t*>(srcData),
                    reinterpret_cast<const int32_t*>(weightsData), reinterpret_cast<int32_t*>(dstData), srcDesc, dstDesc);
        }
        default: {
            IE_THROW() << "EmbeddingBagSum layer does not support precision '"
                        + std::string(srcDesc.getPrecision().name()) + "'";
        }
    }
}
