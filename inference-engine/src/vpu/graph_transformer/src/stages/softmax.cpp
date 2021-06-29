// Copyright (C) 2018-2021 Intel Corporation
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
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<SoftMaxStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();

        auto axis = attrs().get<Dim>("axis");
        auto axisInd = input->desc().dimsOrder().dimInd(axis);

        serializer.append(static_cast<int32_t>(axisInd));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseSoftMax(const Model& model, const NodePtr& _node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];
    const auto& softMax = ngraph::as_type_ptr<ngraph::op::v1::Softmax>(_node);

    IE_ASSERT(softMax != nullptr);

    auto nodeInput = softMax->input(0).get_tensor_ptr();
    IE_ASSERT(nodeInput != nullptr);

    IE_ASSERT(softMax->get_axis() < input->desc().numDims());

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto axis = perm[input->desc().numDims() - 1 - softMax->get_axis()];

    _stageBuilder->addSoftMaxStage(model, softMax->get_name(), softMax, input, output, axis);
}

Stage StageBuilder::addSoftMaxStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        const Data& input,
        const Data& output,
        Dim axis) {
    auto stage = model->addNewStage<SoftMaxStage>(
        name,
        StageType::SoftMax,
        node,
        {input},
        {output});

    stage->attrs().set<Dim>("axis", axis);

    return stage;
}

}  // namespace vpu
