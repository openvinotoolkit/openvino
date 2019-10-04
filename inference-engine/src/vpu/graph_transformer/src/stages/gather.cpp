// Copyright (C) 2018-2019 Intel Corporation
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

void FrontEnd::parseGather(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);
    auto layer = std::dynamic_pointer_cast<ie::GatherLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];

    IE_ASSERT(layer->axis < input->desc().numDims());

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto axis = perm[input->desc().numDims() - 1 - layer->axis];

    _stageBuilder->addGatherStage(model, layer->name, layer, inputs[0], inputs[1], outputs[0], axis);
}

namespace {

class GatherStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<GatherStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step,
            StageDataInfo<float>& scaleInfo) override {
         if (step == ScalePropagationStep::Propagate) {
             scaleInfo.setOutput(outputEdge(0), inputScales[0]);
         } else {
             // Gather can only propagate scaling.
             for (const auto& inEdge : inputEdges()) {
                 scaleInfo.setInput(inEdge, 1.0f);
             }
             scaleInfo.setOutput(outputEdge(0), 1.0f);
         }
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
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
        return StageSHAVEsRequirements::OnlyOne;
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

         input0->serializeNewBuffer(serializer);
         output->serializeNewBuffer(serializer);
         input1->serializeNewBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addGatherStage(
        const Model::Ptr& model,
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
