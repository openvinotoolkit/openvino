// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
#include "ie_parallel.hpp"
#include "list.hpp"

#include <set>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::Extensions::Cpu;

MKLDNNEmbeddingBagSum::MKLDNNEmbeddingBagSum(
            const std::shared_ptr<ngraph::Node>& op,
            size_t requiredInputNum,
            size_t indicesIdx,
            size_t perSampleWeightsIdx,
            size_t defaultIndexIdx,
            const std::set<Precision>& supportedPrecisions) :
                INDICES_IDX(indicesIdx),
                PER_SAMPLE_WEIGHTS_IDX(perSampleWeightsIdx),
                DEFAULT_INDEX_IDX(defaultIndexIdx) {
    try {
        _layerName = op->get_friendly_name();
        std::string logPrefix = std::string("Layer EmbeddingBagSum with name '") + _layerName + "' ";
        if (op->get_input_size() < requiredInputNum || op->get_output_size() != 1)
            IE_THROW() << logPrefix << "has incorrect number of input or output edges!";

        auto inDataPrecision = details::convertPrecision(op->get_input_element_type(EMB_TABLE_IDX));
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
    } catch (InferenceEngine::Exception &ex) {
        errorMsg = ex.what();
        throw;
    }
}

StatusCode MKLDNNEmbeddingBagSum::execute(
            std::vector<Blob::Ptr>& inputs,
            std::vector<Blob::Ptr>& outputs,
            ResponseDesc *resp) noexcept {
    switch (inputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32: {
            return processData<PrecisionTrait<Precision::FP32>::value_type>(inputs, outputs, resp);
        }
        case Precision::I8: {
            return processData<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
        }
        case Precision::U8: {
            return processData<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs, resp);
        }
        case Precision::I32: {
            return processData<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
        }
        default: {
            if (resp) {
                std::string errorMsg = "EmbeddingBagSum layer does not support precision '"
                        + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
    }
}

template<typename T>
StatusCode MKLDNNEmbeddingBagSum::processData(
            std::vector<Blob::Ptr>& inputs,
            std::vector<Blob::Ptr>& outputs,
            ResponseDesc *resp) noexcept {
    std::string errorMsg;
    std::string msgPrefix = std::string("Node EmbeddingBagSum with name '") + _layerName + "' ";

    const T* srcData = inputs[0]->cbuffer().as<const T*>() +
        inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    T* dstData = outputs[0]->buffer().as<T*>() +
        outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

    const T* weightsData = nullptr;
    if (_withWeights)
        weightsData = inputs[PER_SAMPLE_WEIGHTS_IDX]->cbuffer().as<const T*>();
    initFromInputs(inputs);

    const auto& inDataDims = inputs[0]->getTensorDesc().getDims();

    const size_t outputBagsNum = outputs[0]->getTensorDesc().getDims()[0];

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
                    errorMsg = msgPrefix + "' has invalid embedding bag index: " + std::to_string(indices[inIdx]);
                    return;
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
                        errorMsg = msgPrefix + "' has invalid embedding bag index: " + std::to_string(indices[inIdx]);
                        return;
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

    if (!errorMsg.empty()) {
        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        return GENERAL_ERROR;
    }

    return OK;
}
