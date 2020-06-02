// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
#include "ie_parallel.hpp"
#include "jit_generator.hpp"
#include "list.hpp"

#include <set>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::Extensions::Cpu;


const std::set<size_t> MKLDNNEmbeddingBagSum::_supportedIndicesTypeSize = {sizeof(INT32), sizeof(INT64)};

MKLDNNEmbeddingBagSum::MKLDNNEmbeddingBagSum(
            const CNNLayer* layer,
            size_t requiredInputNum,
            size_t indicesIdx,
            size_t perSampleWeightsIdx,
            size_t defaultIndexIdx,
            const std::set<Precision>& supportedPrecisions) :
                INDICES_IDX(indicesIdx),
                PER_SAMPLE_WEIGHTS_IDX(perSampleWeightsIdx),
                DEFAULT_INDEX_IDX(defaultIndexIdx) {
    try {
        std::string logPrefix = std::string("Layer EmbeddingBagSum with name '") + layer->name + "' ";
        if (layer->insData.size() < requiredInputNum || layer->outData.size() != 1)
            THROW_IE_EXCEPTION << logPrefix << "has incorrect number of input or output edges!";
        _layerName = layer->name;

        auto inData = layer->insData[0].lock();
        auto indicesData = layer->insData[INDICES_IDX].lock();
        if (inData == nullptr || indicesData == nullptr)
            THROW_IE_EXCEPTION << logPrefix << "has nullable input data.";

        auto dataPrecision = inData->getTensorDesc().getPrecision();
        if (dataPrecision == Precision::BF16)
            dataPrecision = Precision::FP32;
        if (!supportedPrecisions.empty()) {
            if (supportedPrecisions.find(dataPrecision) == supportedPrecisions.end())
                THROW_IE_EXCEPTION << logPrefix << "has unsupported precision: " << dataPrecision.name();
        } else {
            static const std::set<Precision> defaultSupportedPrecisions =
                {Precision::FP32, Precision::I8, Precision::U8, Precision::I32};
            if (defaultSupportedPrecisions.find(dataPrecision) == defaultSupportedPrecisions.end())
                THROW_IE_EXCEPTION << logPrefix << "has unsupported precision: " << dataPrecision.name();
        }

        if (layer->insData.size() > PER_SAMPLE_WEIGHTS_IDX)
            _withWeights = true;
        if (_withWeights) {
            auto weightsData = layer->insData[PER_SAMPLE_WEIGHTS_IDX].lock();
            if (weightsData == nullptr)
                 THROW_IE_EXCEPTION << logPrefix << "has nullable weights data";
            if (weightsData->getTensorDesc().getDims() != indicesData->getTensorDesc().getDims())
                 THROW_IE_EXCEPTION << logPrefix << "must have equal shapes for indices and per_sample_weights inputs.";
        }

        LayerConfig config;
        config.inConfs.resize(layer->insData.size());
        for (int i = 0; i < layer->insData.size(); i++) {
            auto data = layer->insData[i].lock();
            if (data == nullptr)
                THROW_IE_EXCEPTION << logPrefix << "has nullable input data";
            auto prc = data->getTensorDesc().getPrecision();
            if (prc == Precision::BF16)
                prc = Precision::FP32;
            config.inConfs[i].desc = TensorDesc(prc,
                data->getTensorDesc().getDims(),
                TensorDesc::getLayoutByDims(data->getTensorDesc().getDims()));
        }

        DataConfig outConfig;
        auto& outDims = layer->outData[0]->getTensorDesc().getDims();
        outConfig.desc = TensorDesc(dataPrecision,
            outDims,
            TensorDesc::getLayoutByDims(outDims));
        config.outConfs.push_back(outConfig);
        config.dynBatchSupport = false;

        confs.push_back(config);

        const auto& inDataDims = inData->getTensorDesc().getDims();
        _embDepth = 1lu;
        for (size_t i = 1lu; i < inDataDims.size(); i++) {
            _embDepth *= inDataDims[i];
        }
    } catch (InferenceEngine::details::InferenceEngineException &ex) {
        errorMsg = ex.what();
    }
}

StatusCode MKLDNNEmbeddingBagSum::execute(
            std::vector<Blob::Ptr>& inputs,
            std::vector<Blob::Ptr>& outputs,
            ResponseDesc *resp) noexcept {
    switch (inputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32: {
            processData<PrecisionTrait<Precision::FP32>::value_type>(inputs, outputs);
            break;
        }
        case Precision::I8: {
            processData<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs);
            break;
        }
        case Precision::U8: {
            processData<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
            break;
        }
        case Precision::I32: {
            processData<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
            break;
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

    return OK;
}

template<typename T>
void MKLDNNEmbeddingBagSum::processData(
            std::vector<Blob::Ptr>& inputs,
            std::vector<Blob::Ptr>& outputs) noexcept {
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
        const size_t* indices = nullptr;
        size_t weightsIdx = 0lu;
        bool withWeights = _withWeights;

        for (size_t obi = start; obi < end; obi++) {
            size_t dstIndex = obi * _embDepth;
            getIndices(obi, indices, indicesSize, weightsIdx, withWeights);

            if (indices != nullptr) {
                withWeights = withWeights & _withWeights;

                size_t inIdx = 0lu;
                if (indices[inIdx] >= inDataDims[0])
                    THROW_IE_EXCEPTION << "EmbeddingBagSum layer '" << _layerName
                        << "' has invalid embedding bag index: " << indices[inIdx];
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
                    if (indices[inIdx] >= inDataDims[0])
                        THROW_IE_EXCEPTION << "EmbeddingBagSum layer '" << _layerName
                            << "' has invalid embedding bag index: " << indices[inIdx];
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
