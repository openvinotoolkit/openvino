// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <cmath>

#include <vector>
#include <limits>
#include <memory>
#include <set>
#include <string>

#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class ReLUStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ReLUStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto negativeSlope = attrs().get<float>("negativeSlope");

        serializer.append(static_cast<uint32_t>(numInputs() == 2));
        serializer.append(negativeSlope);
    }
};

}  // namespace

void FrontEnd::parseReLU(const Model& model, const NodePtr& _node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);
    const auto& relu = ngraph::as_type_ptr<ngraph::opset4::Relu>(_node);
    
    IE_ASSERT(relu != nullptr);
    float negativeSlope = 0;
    if (relu->get_input_size() > 1) {
        auto slopeNode = std::dynamic_pointer_cast<ngraph::opset4::Constant>(relu->get_input_node_shared_ptr(1));
        if (slopeNode != nullptr) {
            negativeSlope = slopeNode->cast_vector<float>()[0];
        }
    }
    _stageBuilder->addReLUStage(model, relu->get_name(), relu, negativeSlope, inputs[0], outputs[0]);
}

Stage StageBuilder::addReLUStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        float negativeSlope,
        const Data& input,
        const Data& output,
        const Data& biases) {
    auto stageType = StageType::__SPECIAL_START__;
    if (biases == nullptr) {
        stageType =
            std::fabs(negativeSlope) < std::numeric_limits<float>::epsilon() ?
                StageType::Relu :
                StageType::LeakyRelu;
    } else {
        stageType =
            std::fabs(negativeSlope) < std::numeric_limits<float>::epsilon() ?
                StageType::BiasRelu :
                StageType::BiasLeakyRelu;
    }

    auto stage = model->addNewStage<ReLUStage>(
        name,
        stageType,
        node,
        {input},
        {output});

    if (biases != nullptr) {
        model->addStageInput(stage, biases);
    }

    stage->attrs().set<float>("negativeSlope", negativeSlope);

    return stage;
}

}  // namespace vpu
