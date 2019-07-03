// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

namespace vpu {

namespace {

class SoftMaxStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<SoftMaxStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>&,
            ScalePropagationStep) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<float> out;

        out[input] = 1.0f;
        out[output] = 1.0f;

        return out;
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<DimsOrder> out;

        out[output] = input->desc().dimsOrder();

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        return DataMap<StridesRequirement>();
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        return DataMap<BatchSupport>();
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        auto axis = attrs().get<Dim>("axis");
        auto axisInd = input->desc().dimsOrder().dimInd(axis);

        serializer.append(static_cast<int32_t>(axisInd));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseSoftMax(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    auto layer = std::dynamic_pointer_cast<ie::SoftMaxLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto layerInput = layer->insData[0].lock();
    IE_ASSERT(layerInput != nullptr);

    IE_ASSERT(layer->axis < input->desc().numDims());

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto axis = perm[input->desc().numDims() - 1 - layer->axis];

    _stageBuilder->addSoftMaxStage(model, layer->name, layer, input, output, axis);
}

Stage StageBuilder::addSoftMaxStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        Dim axis) {
    auto stage = model->addNewStage<SoftMaxStage>(
        name,
        StageType::SoftMax,
        layer,
        {input},
        {output});

    stage->attrs().set<Dim>("axis", axis);

    return stage;
}

}  // namespace vpu
