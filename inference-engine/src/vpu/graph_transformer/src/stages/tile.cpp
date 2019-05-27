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

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<float> out;

        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales.at(input);

            out[output] = inputScale;
        } else {
            // Tile can only propagate scaling, not generate.

            out[input] = 1.0f;
            out[output] = 1.0f;
        }

        return out;
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto inOrder = input->desc().dimsOrder();
        auto finalOrder = inOrder;

        DataMap<DimsOrder> out;

        out[input] = finalOrder;
        out[output] = finalOrder;

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        return DataMap<StridesRequirement>();
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        return DataMap<BatchSupport>();
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

    stage->attrs().set<Dim>("axis", axis);
    stage->attrs().set<int>("tiles", layer->tiles);
}

}  // namespace vpu
