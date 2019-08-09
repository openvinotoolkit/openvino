// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

#include <precision_utils.h>

namespace vpu {

namespace {

class LRNStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<LRNStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        _orderInfo.setOutput(_outputEdges[0], input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        // LRN supports both HWC and CHW orders, but requires that input and output have the same stride

        auto reqs = StridesRequirement::compact();
        if (_type == StageType::LRN &&
            input->desc().dimsOrder().dimInd(Dim::C) != 0) {
            reqs.add(1, DimStride::Aligned);
        }

        _stridesInfo.setInput(_inputEdges[0], reqs);
        _stridesInfo.setOutput(_outputEdges[0], reqs);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto size = attrs().get<int>("size");
        auto k = attrs().get<int>("k");
        auto alpha = attrs().get<float>("alpha");
        auto beta = attrs().get<float>("beta");

        serializer.append(static_cast<uint32_t>(size));
        serializer.append(ie::PrecisionUtils::f32tof16(k));
        serializer.append(ie::PrecisionUtils::f32tof16(alpha));
        serializer.append(ie::PrecisionUtils::f32tof16(beta));
        serializer.append(ie::PrecisionUtils::f32tof16(0));  // for alignment
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        input->serializeOldBuffer(handle_from_this(), serializer);
        output->serializeOldBuffer(handle_from_this(), serializer);
    }
};

}  // namespace

void FrontEnd::parseNorm(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::NormLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto stage = model->addNewStage<LRNStage>(
        layer->name,
        layer->_isAcrossMaps ? StageType::LRN : StageType::InnerLRN,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("size", layer->_size);
    stage->attrs().set<int>("k", layer->_k);
    stage->attrs().set<float>("alpha", layer->_alpha);
    stage->attrs().set<float>("beta", layer->_beta);
}

}  // namespace vpu
