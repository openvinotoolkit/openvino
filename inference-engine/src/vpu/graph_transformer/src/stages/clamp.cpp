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

namespace {

class ClampStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ClampStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto min_value = attrs().get<float>("min_value");
        auto max_value = attrs().get<float>("max_value");

        serializer.append(static_cast<float>(min_value));
        serializer.append(static_cast<float>(max_value));
    }
};

}  // namespace

void FrontEnd::parseClamp(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);
    const auto& clamp = ngraph::as_type_ptr<ngraph::op::Clamp>(node);
    IE_ASSERT(clamp != nullptr);

    _stageBuilder->addClampStage(model, clamp->get_name(), clamp, clamp->get_min(),  clamp->get_max(), inputs[0], outputs[0]);
}

Stage StageBuilder::addClampStage(
            const Model& model,
            const std::string& name,
            const NodePtr& node,
            float min,
            float max,
            const Data& input,
            const Data& output) {
        auto stage = model->addNewStage<ClampStage>(
                name,
                StageType::Clamp,
                node,
                {input},
                {output});

        stage->attrs().set<float>("min_value", min);
        stage->attrs().set<float>("max_value", max);

        return stage;
    }


}  // namespace vpu
