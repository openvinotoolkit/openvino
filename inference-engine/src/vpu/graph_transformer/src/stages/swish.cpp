// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class SwishStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<SwishStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto beta = attrs().get<float>("beta");

        serializer.append(static_cast<float>(beta));
    }
};

}  // namespace

void FrontEnd::parseSwish(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto swish = ngraph::as_type_ptr<ngraph::op::v4::Swish>(node);
    VPU_THROW_UNLESS(swish != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    VPU_THROW_UNLESS((inputs.size() == 1),
                     "Swish stage with name %s must have 1 input, "
                     "actually provided %d", swish->get_friendly_name(), inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Swish stage with name %s must have only 1 output, "
                     "actually provided %d", swish->get_friendly_name(), outputs.size());

    auto stage = model->addNewStage<SwishStage>(
        swish->get_friendly_name(), StageType::Swish, swish, inputs, outputs);
    auto betaValue = 1.0f;
    if (swish->input_values().size() == 2) {
            auto betaNode = swish->input_value(1).get_node_shared_ptr();
            auto betaConst = std::dynamic_pointer_cast<ngraph::opset4::Constant>(betaNode);

            VPU_THROW_UNLESS(betaConst != nullptr, "Can't parse node with name %s and type %s: cannot get the beta", node->get_friendly_name(), node->get_type_name());
            VPU_THROW_UNLESS(ngraph::op::util::get_single_value(betaConst, betaValue), "Can't parse node with name %s and type %s: cannot get the beta", node->get_friendly_name(), node->get_type_name());
        }
    stage->attrs().set<float>("beta", betaValue);
}

}  // namespace vpu
