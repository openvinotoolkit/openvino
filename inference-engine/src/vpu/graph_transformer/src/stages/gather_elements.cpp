// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <string>

namespace vpu {

namespace {

class GatherElementsStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<GatherElementsStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder> &orderInfo) override {
        const auto input1 = inputEdge(0)->input();
        const auto input2 = inputEdge(1)->input();
        const auto output = outputEdge(0)->output();

        orderInfo.setInput(inputEdge(0),
                           DimsOrder::fromNumDims(input1->desc().numDims()));
        orderInfo.setInput(inputEdge(1),
                           DimsOrder::fromNumDims(input2->desc().numDims()));
        orderInfo.setOutput(outputEdge(0),
                            DimsOrder::fromNumDims(output->desc().numDims()));
    }

    void getDataStridesRequirementsImpl(
        StageDataInfo<StridesRequirement> &stridesInfo) override {
        for (const auto &inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {}

    void
    getBatchSupportInfoImpl(StageDataInfo<BatchSupport> &batchInfo) override {}

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        VPU_THROW_UNLESS(numInputs() == 2,
                         "{} stage with name {} must have only 1 output, actually "
                         "provided {} inputs",
                         type(), name(), numInputs());
        VPU_THROW_UNLESS(numOutputs() == 1,
                         "{} stage with name {} must have only 1 output, actually "
                         "provided {} outputs",
                         type(), name(), numInputs());
        VPU_THROW_UNLESS(inputs()[0]->desc().type() == outputs()[0]->desc().type(),
                         "First input and output must have the same DataType, "
                         "actual input type is {} and output type is {}",
                         inputs()[0]->desc().type(), outputs()[0]->desc().type());
        assertInputsOutputsTypes(
            this, {{DataType::U8, DataType::FP16, DataType::S32}, {DataType::S32}},
            {{DataType::U8, DataType::FP16, DataType::S32}});
    }

    void serializeParamsImpl(BlobSerializer &serializer) const override {
        const auto axis = attrs().get<int32_t>("axis");
        serializer.append(axis);
    }

    void serializeDataImpl(BlobSerializer &serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input0->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        input1->serializeBuffer(serializer);
    }
};

}// namespace

Stage StageBuilder::addGatherElementsStage(const Model &model,
                                           const std::string &name,
                                           const ie::CNNLayerPtr &layer,
                                           const Data &input, const Data &indices,
                                           const Data &output, int32_t axis) {
    auto stage = model->addNewStage<GatherElementsStage>(
        layer->name, StageType::GatherElements, layer, {input, indices}, {output});

    stage->attrs().set<int32_t>("axis", axis);

    return stage;
}

void FrontEnd::parseGatherElements(const Model &model, const ie::CNNLayerPtr &layer,
                                   const DataVector &inputs,
                                   const DataVector &outputs) const {
    VPU_THROW_UNLESS(layer, "CNNLayer pointer is null.");
    VPU_THROW_UNLESS(inputs.size() == 2,
                     "{} layer with name {} must have 2 inputs, actually "
                     "provided {} inputs",
                     layer->type, layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "{} layer with name {} must have only 1 output, actually "
                     "provided {} outputs",
                     layer->type, layer->name, outputs.size());

    const auto axis = layer->GetParamAsInt("axis");
    const auto rank = inputs[0]->desc().numDims();

    VPU_THROW_UNLESS(rank >= 1, "rank has to be more than or equal to 1, actually {}", rank);
    VPU_THROW_UNLESS(inputs[1]->desc().numDims() == rank, "rank of the second input must be equal to {}, actually {}",
                     rank, inputs[1]->desc().numDims());
    VPU_THROW_UNLESS(outputs[0]->desc().numDims() == rank, "rank of output must be equal to {}, actually {}",
                     rank, outputs[0]->desc().numDims());
    VPU_THROW_UNLESS(axis >= 0 && axis < rank, "axis must be in the range of [0, {}) , actually {}",
                     rank, axis);

    _stageBuilder->addGatherElementsStage(model, layer->name, layer, inputs[0],
                                          inputs[1], outputs[0], axis);
}

}// namespace vpu
