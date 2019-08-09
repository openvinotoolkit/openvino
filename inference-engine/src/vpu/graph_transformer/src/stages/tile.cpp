// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>
#include <utility>

namespace vpu {

namespace {

class TileStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<TileStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales[0];
            _scaleInfo.setOutput(_outputEdges[0], inputScale);
        } else {
            // Tile can only propagate scaling, not generate.
            _scaleInfo.setInput(_inputEdges[0], 1.0f);
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        auto inOrder = input->desc().dimsOrder();
        auto finalOrder = inOrder;

        _orderInfo.setInput(_inputEdges[0], finalOrder);
        _orderInfo.setOutput(_outputEdges[0], finalOrder);
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::OnlyOne;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto axis = attrs().get<Dim>("axis");
        auto tiles = attrs().get<int>("tiles");

        auto axisInd = output->desc().dimsOrder().dimInd(axis);
        IE_ASSERT(axisInd >= 0);

        serializer.append(static_cast<int32_t>(axisInd));
        serializer.append(static_cast<int32_t>(tiles));
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

void FrontEnd::parseTile(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::TileLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    IE_ASSERT(layer->axis < input->desc().numDims());

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto axis = perm[input->desc().numDims() - 1 - layer->axis];

    auto stage = model->addNewStage<TileStage>(
        layer->name,
        StageType::Tile,
        layer,
        {input},
        {output});

    stage->attrs().set("axis", axis);
    stage->attrs().set("tiles", layer->tiles);
}

}  // namespace vpu
