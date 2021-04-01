// Copyright (C) 2018-2020 Intel Corporation
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
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<TileStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        auto inOrder = input->desc().dimsOrder();
        auto finalOrder = inOrder;

        orderInfo.setInput(inputEdge(0), finalOrder);
        orderInfo.setOutput(outputEdge(0), finalOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::OnlyOne;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        auto axis = attrs().get<Dim>("axis");
        auto tiles = attrs().get<int>("tiles");

        auto axisInd = output->desc().dimsOrder().dimInd(axis);
        IE_ASSERT(axisInd >= 0);

        serializer.append(static_cast<int32_t>(axisInd));
        serializer.append(static_cast<int32_t>(tiles));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseTile(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::TileLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    IE_ASSERT(layer->axis < input->desc().numDims());

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto axis = perm[input->desc().numDims() - 1 - layer->axis];

    auto stage = model->addNewStage<TileStage>(layer->name, StageType::Tile, layer, {input}, {output});
    stage->attrs().set("axis", axis);
    stage->attrs().set("tiles", layer->tiles);
}

}  // namespace vpu
