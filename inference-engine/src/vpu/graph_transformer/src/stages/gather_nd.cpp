// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <string>

namespace vpu {

namespace {

class GatherNDStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<GatherNDStage>(*this);
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
                         "{} stage with name {} must have 2 inputs, actually "
                         "provided {} inputs",
                         type(), name(), numInputs());
        VPU_THROW_UNLESS(numOutputs() == 1,
                         "{} stage with name {} must have 1 output, actually "
                         "provided {} outputs",
                         type(), name(), numOutputs());
        VPU_THROW_UNLESS(inputs()[0]->desc().type() == outputs()[0]->desc().type(),
                         "First input and output must have the same DataType, "
                         "actual input type is {} and output type is {}",
                         inputs()[0]->desc().type(), outputs()[0]->desc().type());
        assertInputsOutputsTypes(
            this, {{DataType::U8, DataType::FP16, DataType::S32}, {DataType::S32}},
            {{DataType::U8, DataType::FP16, DataType::S32}});
    }

    void serializeParamsImpl(BlobSerializer &serializer) const override {
        const auto batchDims = attrs().get<int32_t>("batch_dims");
        serializer.append(batchDims);
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

Stage StageBuilder::addGatherNDStage(const Model& model,
                                     const std::string& name,
                                     const NodePtr& node,
                                     const Data& input, const Data& indices,
                                     const Data& output, int32_t batchDims) {
    auto stage = model->addNewStage<GatherNDStage>(
        node->get_name(), StageType::GatherND, node, {input, indices}, {output});

    stage->attrs().set<int32_t>("batch_dims", batchDims);

    return stage;
}

void FrontEnd::parseGatherND(const Model &model, const NodePtr& node,
                             const DataVector &inputs,
                             const DataVector &outputs) const {
    auto gatherND = ngraph::as_type_ptr<ngraph::op::v5::GatherND>(node);                                 
    VPU_THROW_UNLESS(gatherND != nullptr, "Node pointer is null.");
    VPU_THROW_UNLESS(inputs.size() == 2,
                     "{} layer with name {} must have 2 inputs, actually "
                     "provided {} inputs",
                     gatherND->get_type_name(), gatherND->get_name(), inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "{} layer with name {} must have 1 output, actually "
                     "provided {} outputs",
                     gatherND->get_type_name(), gatherND->get_name(), outputs.size());

    const auto batchDims = gatherND->get_batch_dims();

    _stageBuilder->addGatherNDStage(model, gatherND->get_name(), gatherND, inputs[0],
                                    inputs[1], outputs[0], batchDims);
}

}// namespace vpu
