// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
#include "common/cpu_memcpy.h"
#include <ngraph/op/embedding_segments_sum.hpp>
#include <nodes/common/tensor_desc_creator.h>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class EmbeddingSegmentsSumImpl: public MKLDNNEmbeddingBagSum {
public:
    bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto embBagSegSumOp = ngraph::as_type_ptr<const ngraph::op::v3::EmbeddingSegmentsSum>(op);
            if (!embBagSegSumOp) {
                errorMessage = "Node is not an instance of the EmbeddingSegmentsSum operation from opset v3.";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    explicit EmbeddingSegmentsSumImpl(const std::shared_ptr<ngraph::Node>& op) :
                MKLDNNEmbeddingBagSum(op, 4lu, 1lu, 5lu, 4lu) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            std::string errPrefix = std::string("EmbeddingSegmentsSum layer with name '") + _layerName + "' ";
            if (op->get_input_shape(INDICES_IDX).size() != 1)
                IE_THROW() << errPrefix << "has indices data with invalid shape: "
                    << op->get_input_shape(INDICES_IDX).size();

            if (op->get_input_shape(SEGMENT_ID_IDX).size() != 1)
                IE_THROW() << errPrefix << "has invalid segmentID data shape: "
                    << op->get_input_shape(SEGMENT_ID_IDX).size();

            Precision inDataPrecision = details::convertPrecision(op->get_input_element_type(EMB_TABLE_IDX));

            std::vector<DataConfigurator> inDataConfigurators({
                    {TensorDescCreatorTypes::ncsp, inDataPrecision},
                    {TensorDescCreatorTypes::ncsp, Precision::I32},
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
        indices_ = inputs[INDICES_IDX]->cbuffer().as<const int*>();
        indicesSize_ = inputs[INDICES_IDX]->size();

        segmentIds_ = inputs[SEGMENT_ID_IDX]->cbuffer().as<const int*>();

        if (inputs.size() > NUM_SEGMENTS_IDX) {
            numSegments_ = inputs[NUM_SEGMENTS_IDX]->cbuffer().as<const int*>()[0];
        }

        if (inputs.size() > DEFAULT_INDEX_IDX) {
            defaultIndices_ = inputs[DEFAULT_INDEX_IDX]->cbuffer().as<const int*>();
        }
    }

    void getIndices(int embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) override {
        if (embIndex >= numSegments_)
            IE_THROW() << "Invalid embedding bag index.";

        indices = nullptr;
        size = 0;
        withWeight = true;

        for (int si = 0; si < indicesSize_; si++) {
            if (segmentIds_[si] == embIndex) {
                size++;
                if (indices == nullptr) {
                    indices = indices_ + si;
                    weightsIdx = si;
                }
            }
        }

        // Empty bag
        if (size == 0) {
            size = 1lu;
            withWeight = false;
            if (defaultIndices_)
                indices = defaultIndices_;
            return;
        }
    }

protected:
    const size_t SEGMENT_ID_IDX = 2lu;
    const size_t NUM_SEGMENTS_IDX = 3lu;

    int numSegments_ = 0;

    const int* indices_;
    const int* segmentIds_;
    const int* defaultIndices_ = nullptr;

    size_t indicesSize_ = 0;
};

REG_FACTORY_FOR(EmbeddingSegmentsSumImpl, EmbeddingSegmentsSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
