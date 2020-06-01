// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
#include "ie_parallel.hpp"

#include <vector>


namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class EmbeddingBagOffsetsSumImpl: public MKLDNNEmbeddingBagSum {
public:
    explicit EmbeddingBagOffsetsSumImpl(const CNNLayer* layer) :
            MKLDNNEmbeddingBagSum(layer, 3lu, 1lu, 4lu, 3lu) {
        auto indicesData = layer->insData[INDICES_IDX].lock();
        if (indicesData == nullptr)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has nullable indices data.";
        if (indicesData->getTensorDesc().getDims().size() != 1)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has indices data with invalid shape.";

        auto offsetsData = layer->insData[OFFSETS_IDX].lock();
        if (offsetsData == nullptr)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has invalid offsets data.";
        if (offsetsData->getTensorDesc().getDims().size() != 1)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer's offsets data has invalid shape.";

        _indices_len = indicesData->getTensorDesc().getDims()[0];
        _offsets_len = offsetsData->getTensorDesc().getDims()[0];
    }

    StatusCode execute(
            std::vector<Blob::Ptr>& inputs,
            std::vector<Blob::Ptr>& outputs,
            ResponseDesc* resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision()) {
            case Precision::FP32: {
                return process_data<PrecisionTrait<Precision::FP32>::value_type>(inputs, outputs, resp);
            }
            case Precision::I8: {
                return process_data<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
            }
            case Precision::U8: {
                return process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs, resp);
            }
            case Precision::I32: {
                return process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            }
            default: {
                if (resp) {
                    std::string errorMsg = "EmbeddingBagSum layer does not support embedding table precision '"
                            + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
    }

protected:
    template<typename T>
    StatusCode process_data(
                std::vector<Blob::Ptr>& inputs,
                std::vector<Blob::Ptr>& outputs,
                ResponseDesc* resp) noexcept {
        switch (inputs[1]->getTensorDesc().getPrecision()) {
            case Precision::I32: {
                return process_data<T, PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            }
            case Precision::I64: {
                return process_data<T, PrecisionTrait<Precision::I64>::value_type>(inputs, outputs, resp);
            }
            case Precision::U64: {
                return process_data<T, PrecisionTrait<Precision::U64>::value_type>(inputs, outputs, resp);
            }
            default: {
                if (resp) {
                    std::string errorMsg = "EmbeddingBagSum layer does not support indices precision '"
                            + std::string(inputs[1]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
    }

    template<typename T, typename I>
    StatusCode process_data(
                std::vector<Blob::Ptr>& inputs,
                std::vector<Blob::Ptr>& outputs,
                ResponseDesc* resp) noexcept {
        std::string error_msg;

        const T* src_data = inputs[0]->cbuffer().as<const T*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dst_data = outputs[0]->buffer().as<T*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const I* indices_data = inputs[INDICES_IDX]->cbuffer().as<const I*>();

        const I* offsets_data = inputs[OFFSETS_IDX]->cbuffer().as<const I*>();
        int64_t default_index = -1;
        if (inputs.size() > DEFAULT_INDEX_IDX) {
            default_index = (int64_t)inputs[DEFAULT_INDEX_IDX]->cbuffer().as<const I*>()[0];
            if (default_index < 0 || default_index >= _indices_len) {
                std::snprintf(resp->msg, sizeof(resp->msg), "Invalid default index: %ld\n", default_index);
                return GENERAL_ERROR;
            }
        }
        const T* weights_data = nullptr;
        if (_with_weights)
            weights_data = inputs[PER_SAMPLE_WEIGHTS_IDX]->cbuffer().as<const T*>();

        const auto& inDataDims = inputs[0]->getTensorDesc().getDims();
        const size_t IN_BAGS_DEPTH = inDataDims.size() - 1;

        const size_t OUTPUT_BAGS_NUM = outputs[0]->getTensorDesc().getDims()[0];

        std::function<void(size_t, const I*&, size_t&, size_t&, bool&)> get_idx =
                [&](size_t emb_index, const I*& indices_ref, size_t& out_size, size_t& weights_idx, bool& with_weights) {
            if (emb_index >= _offsets_len) {
                error_msg = std::string("Layer EmbeddingBagOffsetsSum with name '") + _l_name
                    + "' has invalid embedding bag index.";
                return;
            }
            if (offsets_data[emb_index] >= _indices_len) {
                error_msg = std::string("Layer EmbeddingBagOffsetsSum with name '") + _l_name
                    + "'. Offset value exceeds indices size in the model.\noffset: "
                    + std::to_string(offsets_data[emb_index]) + "; indices size: " + std::to_string(_indices_len);
                return;
            }

            indices_ref = nullptr;
            out_size = 0lu;
            with_weights = _with_weights;

            if (emb_index == _offsets_len - 1lu)
                out_size = _indices_len - offsets_data[emb_index];
            else
                out_size = offsets_data[emb_index + 1lu] - offsets_data[emb_index];

            if (out_size != 0lu) {
                indices_ref = indices_data + offsets_data[emb_index];
            } else {
            // Empty or default bag
                with_weights = false;
                if (default_index >= 0) {
                    indices_ref = reinterpret_cast<I*>(&default_index);
                    out_size = 1lu;
                }
                return;
            }

            if (with_weights)
                weights_idx = offsets_data[emb_index];
        };

        auto thread_body = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(OUTPUT_BAGS_NUM, nthr, ithr, start, end);
            if (start >= end)
                return;

            size_t indices_size = 0lu;
            const I* indices = nullptr;
            size_t weights_idx = 0lu;
            bool with_weights = _with_weights;

            for (size_t obi = start; obi < end; obi++) {
                size_t dst_index = obi * _embDepth;
                get_idx(obi, indices, indices_size, weights_idx, with_weights);
                if (indices != nullptr) {
                    with_weights = with_weights & _with_weights;

                    size_t in_idx = 0lu;
                    if (indices[in_idx] >= inDataDims[0]) {
                        error_msg = std::string("EmbeddingBagSum layer '") + _l_name
                            + "' has invalid embedding bag index: " + std::to_string(indices[in_idx]);
                        return;
                    }
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
                        if (indices[in_idx] >= inDataDims[0]) {
                            error_msg = std::string("EmbeddingBagSum layer '") + _l_name
                                + "' has invalid embedding bag index: " + std::to_string(indices[in_idx]);
                            return;
                        }
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

        if (!error_msg.empty()) {
            error_msg.copy(resp->msg, sizeof(resp->msg) - 1);
            return GENERAL_ERROR;
        }

        return OK;
    }

    void init_from_inputs(std::vector<Blob::Ptr>& inputs) override {
    }

    void get_indices(size_t emb_index, const size_t*& indices, size_t& size, size_t& weights_idx, bool& with_weights) override {
    }

    const size_t OFFSETS_IDX = 2lu;

    size_t _indices_len;
    size_t _offsets_len;
};

REG_FACTORY_FOR(EmbeddingBagOffsetsSumImpl, EmbeddingBagOffsetsSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
