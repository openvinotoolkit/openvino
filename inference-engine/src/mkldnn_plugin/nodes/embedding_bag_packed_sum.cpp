// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
#include "common/cpu_memcpy.h"
#include <ngraph/op/embeddingbag_packedsum.hpp>
#include <nodes/common/tensor_desc_creator.h>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class EmbeddingBagPackedSumImpl: public MKLDNNEmbeddingBagSum {
public:
    bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto embBagPackedSumOp = ngraph::as_type_ptr<const ngraph::op::v3::EmbeddingBagPackedSum>(op);
            if (!embBagPackedSumOp) {
                errorMessage = "Node is not an instance of the EmbeddingBagPackedSum operation from opset v3.";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    explicit EmbeddingBagPackedSumImpl(const std::shared_ptr<ngraph::Node>& op) :
            MKLDNNEmbeddingBagSum(op, 2lu, 1lu, 2lu, 3lu) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            if (op->get_input_shape(INDICES_IDX).size() != 2)
                IE_THROW() << "'" << _layerName << "' layer has indices data with invalid shape.";
            _batch = op->get_input_shape(INDICES_IDX)[0];
            _indicesPerBag = op->get_input_shape(INDICES_IDX)[1];

            Precision inDataPrecision = details::convertPrecision(op->get_input_element_type(EMB_TABLE_IDX));

            std::vector<DataConfigurator> inDataConfigurators({
                    {TensorDescCreatorTypes::ncsp, inDataPrecision},
                    {TensorDescCreatorTypes::ncsp, Precision::I32}});
            if (op->get_input_size() > PER_SAMPLE_WEIGHTS_IDX)
                inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp, inDataPrecision});

            addConfig(op, inDataConfigurators, {{TensorDescCreatorTypes::ncsp, inDataPrecision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    void initFromInputs(std::vector<Blob::Ptr>& inputs) override {
        _indices = inputs[INDICES_IDX]->cbuffer().as<const int*>();
    }

    void getIndices(int embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeights) override {
        if (embIndex >= _batch * _indicesPerBag)
            IE_THROW() << "Invalid embedding bag index.";

        withWeights = true;

        indices = _indices + embIndex * _indicesPerBag;
        size = _indicesPerBag;

        weightsIdx = embIndex * _indicesPerBag;
    }

protected:
    const int* _indices;
    size_t _batch = 0;
    size_t _indicesPerBag = 0;
};

REG_FACTORY_FOR(EmbeddingBagPackedSumImpl, EmbeddingBagPackedSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
