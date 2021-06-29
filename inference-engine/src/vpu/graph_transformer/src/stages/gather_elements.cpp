// Copyright (C) 2018-2021 Intel Corporation
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

        const auto rowIndicesMode = attrs().get<int32_t>("rowIndicesMode");
        if (rowIndicesMode) {
            const auto input3 = inputEdge(2)->input();
            orderInfo.setInput(inputEdge(2),
                               DimsOrder::fromNumDims(input3->desc().numDims()));
        }

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
        const auto axis = attrs().get<int32_t>("axis");
        const auto rank = inputEdge(0)->input()->desc().numDims();
        const auto rowIndicesMode = attrs().get<int32_t>("rowIndicesMode");

        return (rowIndicesMode || (axis == rank - 1)) ? StageSHAVEsRequirements::NeedMax : StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        VPU_THROW_UNLESS(numInputs() == 2 || numInputs() == 3,
                         "{} stage with name {} must have 2 or 3 inputs only, actually "
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

        DataTypesRequirement inputDataTypes = {{DataType::U8, DataType::FP16, DataType::S32}, {DataType::S32}};
        if (numInputs() == 3)
            inputDataTypes.push_back({DataType::S32});

        assertInputsOutputsTypes(this, inputDataTypes, {{DataType::U8, DataType::FP16, DataType::S32}});
    }

    void serializeParamsImpl(BlobSerializer &serializer) const override {
        const auto axis = attrs().get<int32_t>("axis");
        const auto rowIndicesMode = attrs().get<int32_t>("rowIndicesMode");
        serializer.append(axis);
        serializer.append(rowIndicesMode);
    }

    void serializeDataImpl(BlobSerializer &serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input0->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        input1->serializeBuffer(serializer);

        const auto rowIndicesMode = attrs().get<int32_t>("rowIndicesMode");

        if (rowIndicesMode) {
            auto rowIndices = inputEdge(2)->input();
            rowIndices->serializeBuffer(serializer);
        }
    }
};

}// namespace

Stage StageBuilder::addGatherElementsStage(const Model& model,
                                           const std::string& name,
                                           const NodePtr& node,
                                           const DataVector& inputs,
                                           const Data& output, int32_t axis,
                                           bool rowIndicesMode) {
    auto stage = model->addNewStage<GatherElementsStage>(
        node->get_name(), StageType::GatherElements, node, inputs, {output});

    stage->attrs().set<int32_t>("axis", axis);
    stage->attrs().set<int32_t>("rowIndicesMode", rowIndicesMode);

    return stage;
}

void FrontEnd::parseGatherElements(const Model &model, const NodePtr& node,
                                   const DataVector &inputs,
                                   const DataVector &outputs) const {
    auto gatherElements = ngraph::as_type_ptr<ngraph::op::v6::GatherElements>(node);
    VPU_THROW_UNLESS(gatherElements != nullptr, "Node pointer is null.");
    VPU_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                     "{} layer with name {} must have 2 inputs, actually "
                     "provided {} inputs",
                     gatherElements->get_type_name(), gatherElements->get_name(), inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "{} layer with name {} must have only 1 output, actually "
                     "provided {} outputs",
                     gatherElements->get_type_name(), gatherElements->get_name(), outputs.size());

    bool rowIndicesMode = (inputs.size() == 3);

    const auto axis = gatherElements->get_axis();
    const auto rank = inputs[0]->desc().numDims();

    VPU_THROW_UNLESS(rank >= 1, "rank has to be more than or equal to 1, actually {}", rank);

    if (rowIndicesMode) {
        VPU_THROW_UNLESS(inputs[1]->desc().numDims() == rank + 1, "rank of the second input must be equal to {}, actually {}",
                        rank + 1, inputs[1]->desc().numDims());
        VPU_THROW_UNLESS(inputs[2]->desc().numDims() == 2, "rank of the third input must be equal to 2, actually {}",
                        2, inputs[2]->desc().numDims());
        VPU_THROW_UNLESS(outputs[0]->desc().numDims() == rank + 1, "rank of output must be equal to {}, actually {}",
                        rank + 1, outputs[0]->desc().numDims());
        VPU_THROW_UNLESS(axis == rank - 1, "axis must be equal to {}, actually {}", rank - 1, axis);
    } else {
        VPU_THROW_UNLESS(inputs[1]->desc().numDims() == rank, "rank of the second input must be equal to {}, actually {}",
                        rank, inputs[1]->desc().numDims());
        VPU_THROW_UNLESS(outputs[0]->desc().numDims() == rank, "rank of output must be equal to {}, actually {}",
                        rank, outputs[0]->desc().numDims());
        VPU_THROW_UNLESS(axis >= 0 && axis < rank, "axis must be in the range of [0, {}) , actually {}",
                        rank, axis);
    }

    _stageBuilder->addGatherElementsStage(model, gatherElements->get_name(), gatherElements, inputs, outputs[0], axis, rowIndicesMode);
}

}// namespace vpu
