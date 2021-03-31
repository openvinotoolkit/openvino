// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
#include "ie_parallel.hpp"
#include <ngraph/op/embeddingbag_offsets_sum.hpp>
#include <nodes/common/tensor_desc_creator.h>

#include <vector>


namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class EmbeddingBagOffsetsSumImpl: public MKLDNNEmbeddingBagSum {
public:
    bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto embBagOffsetSumOp = ngraph::as_type_ptr<const ngraph::op::v3::EmbeddingBagOffsetsSum>(op);
            if (!embBagOffsetSumOp) {
                errorMessage = "Node is not an instance of the EmbeddingBagOffsetsSum operation from opset v3.";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    explicit EmbeddingBagOffsetsSumImpl(const std::shared_ptr<ngraph::Node>& op) :
                MKLDNNEmbeddingBagSum(op, 3lu, 1lu, 4lu, 3lu) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            if (op->get_input_shape(INDICES_IDX).size() != 1)
                IE_THROW() << "'" << _layerName << "' layer has indices data with invalid shape.";

            if (op->get_input_shape(OFFSETS_IDX).size() != 1)
                IE_THROW() << "'" << _layerName << "' layer's offsets data has invalid shape.";

            _indicesLen = op->get_input_shape(INDICES_IDX)[0];
            _offsetsLen = op->get_input_shape(OFFSETS_IDX)[0];

            Precision inDataPrecision = details::convertPrecision(op->get_input_element_type(EMB_TABLE_IDX));

            std::vector<DataConfigurator> inDataConfigurators({
                    {TensorDescCreatorTypes::ncsp, inDataPrecision},
                    {TensorDescCreatorTypes::ncsp, Precision::I32},
                    {TensorDescCreatorTypes::ncsp, Precision::I32}});
            if (op->get_input_size() > DEFAULT_INDEX_IDX)
                inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp, Precision::I32});
            if (op->get_input_size() > PER_SAMPLE_WEIGHTS_IDX)
                inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp, inDataPrecision});

            addConfig(op, inDataConfigurators, {{TensorDescCreatorTypes::ncsp, inDataPrecision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    void initFromInputs(std::vector<Blob::Ptr>& inputs) override {
        indicesData_ = inputs[INDICES_IDX]->cbuffer().as<const int*>();
        offsetsData_ = inputs[OFFSETS_IDX]->cbuffer().as<const int*>();

        if (inputs.size() > DEFAULT_INDEX_IDX) {
            defaultIndices_ = inputs[DEFAULT_INDEX_IDX]->cbuffer().as<const int*>();
        }
    }

    void getIndices(int embIndex, const int*& indicesRef, size_t& outSize, int& weightsIdx, bool& withWeights) override {
        if (embIndex >= _offsetsLen) {
            IE_THROW() << "Invalid embedding bag index.";
        }
        if (offsetsData_[embIndex] >= _indicesLen) {
            IE_THROW() << "Offset value exceeds indices size.";
        }

        indicesRef = nullptr;
        outSize = 0lu;
        withWeights = _withWeights;

        if (embIndex == _offsetsLen - 1lu)
            outSize = _indicesLen - offsetsData_[embIndex];
        else
            outSize = offsetsData_[embIndex + 1lu] - offsetsData_[embIndex];

        if (outSize != 0lu) {
            indicesRef = indicesData_ + offsetsData_[embIndex];
        } else {
        // Empty or default bag
            withWeights = false;
            if (defaultIndices_) {
                indicesRef = defaultIndices_;
                outSize = 1lu;
            }
            return;
        }

        if (withWeights)
            weightsIdx = offsetsData_[embIndex];
    }

    const size_t OFFSETS_IDX = 2lu;

    const int* indicesData_ = nullptr;
    const int* offsetsData_ = nullptr;
    const int* defaultIndices_ = nullptr;

    size_t _indicesLen;
    size_t _offsetsLen;
};

REG_FACTORY_FOR(EmbeddingBagOffsetsSumImpl, EmbeddingBagOffsetsSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
