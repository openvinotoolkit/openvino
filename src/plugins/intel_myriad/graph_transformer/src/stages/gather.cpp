// Copyright (C) 2018-2022 Intel Corporation
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

void FrontEnd::parseGather(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(layer != nullptr, "Encountered nullptr CNN layer");
    VPU_THROW_UNLESS(inputs.size() == 3, "Expected {} inputs (data, indices, axis), got {}", 3, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1, "Expected {} outputs, got {}", 1, outputs.size());

    VPU_THROW_UNLESS(inputs[2]->usage() == DataUsage::Const, "Only constant axis is supported, but got {} data object", inputs[2]->usage());
    VPU_THROW_UNLESS(inputs[2]->desc().type() == DataType::S32, "Only {} is supported as axis data type, got {}", DataType::S32, inputs[2]->desc().type());
    VPU_THROW_UNLESS(inputs[2]->desc().numDims() == 1, "Only single value axis is supported, got {}D data object", inputs[2]->desc().numDims());
    VPU_THROW_UNLESS(inputs[2]->desc().totalDimSize() == 1, "Only single value axis is supported, got {} elements", inputs[2]->desc().totalDimSize());

    auto input = inputs[0];
    const auto axis = inputs[2]->content()->get<std::int32_t>()[0];
    const auto ieNormalizedAxis = axis < 0 ? input->desc().numDims() + axis : axis;
    VPU_THROW_UNLESS(ieNormalizedAxis >= 0 && ieNormalizedAxis < input->desc().numDims(),
        "Axis value must fit into input tensor, got axis = {}, input rank = {}", axis, input->desc().numDims());

    const auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
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
