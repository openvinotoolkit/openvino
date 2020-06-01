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


const std::set<size_t> MKLDNNEmbeddingBagSum::_supported_indexes_type_size = {sizeof(INT32), sizeof(INT64)};

MKLDNNEmbeddingBagSum::MKLDNNEmbeddingBagSum(
        const CNNLayer* layer,
        size_t required_input_num,
        size_t indices_idx,
        size_t per_sample_weights_idx,
        size_t default_index_idx,
        const std::set<Precision>& supported_precisions) :
            INDICES_IDX(indices_idx),
            PER_SAMPLE_WEIGHTS_IDX(per_sample_weights_idx),
            DEFAULT_INDEX_IDX(default_index_idx) {
    try {
        std::string log_prefix = "Layer EmbeddingBagSum with name '";
        if (layer->insData.size() < required_input_num || layer->outData.size() != 1)
            THROW_IE_EXCEPTION << log_prefix << layer->name << "' has incorrect number of input or output edges!";
        _l_name = layer->name;

        auto inData = layer->insData[0].lock();
        auto indicesData = layer->insData[INDICES_IDX].lock();
        if (inData == nullptr || indicesData == nullptr)
            THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has nullable input data.";

        const auto precision = inData->getTensorDesc().getPrecision();
        if (!supported_precisions.empty()) {
            if (supported_precisions.find(precision) == supported_precisions.end())
                THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has unsupported precision: " << precision.name();
        } else {
            static const std::set<Precision> default_supported_precisions =
                {Precision::FP32, Precision::BF16, Precision::I8, Precision::U8, Precision::I32};
            if (default_supported_precisions.find(precision) == default_supported_precisions.end())
                THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has unsupported precision: " << precision.name();
        }

        if (layer->insData.size() > PER_SAMPLE_WEIGHTS_IDX)
            _with_weights = true;
        if (_with_weights) {
            auto weightsData = layer->insData[PER_SAMPLE_WEIGHTS_IDX].lock();
            if (weightsData == nullptr)
                 THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has nullable weights data";
            if (weightsData->getTensorDesc().getDims() != indicesData->getTensorDesc().getDims())
                 THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer must have equal shapes for indices and per_sample_weights inputs.";
        }

        LayerConfig config;
        config.inConfs.resize(layer->insData.size());
        for (int i = 0; i < layer->insData.size(); i++) {
            auto data = layer->insData[i].lock();
            if (data == nullptr)
                THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has nullable input data";
            config.inConfs[i].desc = TensorDesc(data->getTensorDesc());
        }

        DataConfig outConfig;
        outConfig.desc = TensorDesc(inData->getTensorDesc().getPrecision(),
            layer->outData[0]->getTensorDesc().getDims(),
            layer->outData[0]->getTensorDesc().getLayout());
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
        case Precision::FP32:
        case Precision::BF16: {
            process_data<PrecisionTrait<Precision::FP32>::value_type>(inputs, outputs);
            break;
        }
        case Precision::I8: {
            process_data<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs);
            break;
        }
        case Precision::U8: {
            process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
            break;
        }
        case Precision::I32: {
            process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
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
void MKLDNNEmbeddingBagSum::process_data(
        std::vector<Blob::Ptr>& inputs,
        std::vector<Blob::Ptr>& outputs) noexcept {
    const T* src_data = inputs[0]->cbuffer().as<const T*>() +
        inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    T* dst_data = outputs[0]->buffer().as<T*>() +
        outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    const T* weights_data = nullptr;
    if (_with_weights)
        weights_data = inputs[PER_SAMPLE_WEIGHTS_IDX]->cbuffer().as<const T*>();
    init_from_inputs(inputs);

    const auto& inDataDims = inputs[0]->getTensorDesc().getDims();

    const size_t OUTPUT_BAGS_NUM = outputs[0]->getTensorDesc().getDims()[0];

    auto thread_body = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(OUTPUT_BAGS_NUM, nthr, ithr, start, end);
        if (start >= end)
            return;

        size_t indices_size = 0lu;
        const size_t* indices = nullptr;
        size_t weights_idx = 0lu;
        bool with_weights = _with_weights;

        for (size_t obi = start; obi < end; obi++) {
            size_t dst_index = obi * _embDepth;
            get_indices(obi, indices, indices_size, weights_idx, with_weights);

            if (indices != nullptr) {
                with_weights = with_weights & _with_weights;

                size_t in_idx = 0lu;
                if (indices[in_idx] >= inDataDims[0])
                    THROW_IE_EXCEPTION << "EmbeddingBagSum layer '" << _l_name
                        << "' has invalid embedding bag index: " << indices[in_idx];
                size_t src_index = indices[in_idx] * _embDepth;

                if (with_weights) {
                    for (size_t i = 0lu; i < _embDepth; i++) {
                        dst_data[dst_index + i] = src_data[src_index + i] * weights_data[weights_idx];
                    }
                    weights_idx++;
                } else {
                    for (size_t i = 0lu; i < _embDepth; i++) {
                        dst_data[dst_index + i] = src_data[src_index + i];
                    }
                }

                for (in_idx = 1lu; in_idx < indices_size; in_idx++) {
                    if (indices[in_idx] >= inDataDims[0])
                        THROW_IE_EXCEPTION << "EmbeddingBagSum layer '" << _l_name
                            << "' has invalid embedding bag index: " << indices[in_idx];
                    size_t src_index = indices[in_idx] * _embDepth;

                    if (with_weights) {
                        for (size_t i = 0lu; i < _embDepth; i++) {
                            dst_data[dst_index + i] += src_data[src_index + i] * weights_data[weights_idx];
                        }
                        weights_idx++;
                    } else {
                        for (size_t i = 0lu; i < _embDepth; i++) {
                            dst_data[dst_index + i] += src_data[src_index + i];
                        }
                    }
                }
            } else {
                for (size_t i = 0lu; i < _embDepth; i++) {
                    dst_data[dst_index + i] = 0;
                }
            }
        }
    };

    parallel_nt(0, thread_body);
}
