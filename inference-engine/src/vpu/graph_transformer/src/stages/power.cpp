// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>
#include <string>

#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

void FrontEnd::parsePower(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];
    auto power = ngraph::as_type_ptr<ngraph::opset4::Power>(node);
    VPU_THROW_UNLESS(power != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto pow = 0.f;
    // try to get the pow parameter
    auto powerInput = power->input(1).get_source_output().get_node_shared_ptr();
    auto constNode = ngraph::as_type_ptr<ngraph::opset4::Constant>(powerInput);
    VPU_THROW_UNLESS(constNode != nullptr, "Can't parse node with name %s and type %s. Node with data is nullptr", power->get_friendly_name(), power->get_type_name());
    pow = constNode->get_vector<float>()[0]; // not sure
    
    _stageBuilder->addPowerStage(model, power->get_friendly_name(), power, 1.0f, pow, 0.0f, inputs[0], outputs[0]);
}

void FrontEnd::parseSqrt(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];
    auto sqrt = ngraph::as_type_ptr<ngraph::opset4::Sqrt>(node);
    
    VPU_THROW_UNLESS(sqrt != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    
    _stageBuilder->addPowerStage(model, sqrt->get_friendly_name(), sqrt, 1.0f, 0.5f, 0.0f, inputs[0], outputs[0]);
}

namespace {

class PowerStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PowerStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto scale = attrs().get<float>("scale");
        auto power = attrs().get<float>("power");
        auto bias = attrs().get<float>("bias");

        serializer.append(static_cast<float>(bias));
        serializer.append(static_cast<float>(scale));
        serializer.append(static_cast<float>(power));
    }
};

}  // namespace

Stage StageBuilder::addPowerStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        float scale,
        float power,
        float bias,
        const Data& input,
        const Data& output) {
    auto stage = model->addNewStage<PowerStage>(
        name,
        StageType::Power,
        node,
        {input},
        {output});

    stage->attrs().set<float>("scale", scale);
    stage->attrs().set<float>("power", power);
    stage->attrs().set<float>("bias", bias);

    return stage;
}

}  // namespace vpu
