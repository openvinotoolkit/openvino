// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <list>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

void FrontEnd::parseGather(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);
    auto layer = std::dynamic_pointer_cast<ie::GatherLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];

    IE_ASSERT(layer->axis < input->desc().numDims());

    const auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    const auto ieNormalizedAxis = layer->axis < 0 ? input->desc().numDims() + layer->axis : layer->axis;
    const auto axisDim = perm[input->desc().numDims() - 1 - ieNormalizedAxis];

    _stageBuilder->addGatherStage(model, layer->name, layer, inputs[0], inputs[1], outputs[0], axisDim);
}

namespace {

class GatherStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<GatherStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input1 = inputEdge(0)->input();
        auto input2 = inputEdge(1)->input();
        auto output = outputEdge(0)->output();
        orderInfo.setInput(inputEdge(0), DimsOrder::fromNumDims(input1->desc().numDims()));
        orderInfo.setInput(inputEdge(1), DimsOrder::fromNumDims(input2->desc().numDims()));
        orderInfo.setOutput(outputEdge(0), DimsOrder::fromNumDims(output->desc().numDims()));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
         for (const auto& inEdge : inputEdges()) {
             stridesInfo.setInput(inEdge, StridesRequirement::compact());
         }
         stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        const auto& srcType = input(0)->desc().type();
        assertInputsOutputsTypes(this, {{srcType}, {DataType::FP16, DataType::S32}}, {{srcType}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
         auto input = inputEdge(0)->input();

         auto axis = attrs().get<Dim>("axis");
         auto axisInd = input->desc().dimsOrder().dimInd(axis);

         serializer.append(static_cast<int>(axisInd));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
         auto input0 = inputEdge(0)->input();
         auto input1 = inputEdge(1)->input();
         auto output = outputEdge(0)->output();

         input0->serializeBuffer(serializer);
         output->serializeBuffer(serializer);
         input1->serializeBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addGatherStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input0,
        const Data& input1,
        const Data& output,
        Dim axis) {
    auto stage = model->addNewStage<GatherStage>(
        name,
        StageType::Gather,
        layer,
        {input0, input1},
        {output});

    stage->attrs().set<Dim>("axis", axis);

    return stage;
}

}  // namespace vpu
