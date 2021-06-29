// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/post_op_stage.hpp>

#include <vector>
#include <memory>
#include <set>

namespace vpu {

namespace {

class RoundStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<RoundStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto mode = attrs().get<RoundMode>("mode");
        serializer.append(static_cast<int>(mode));
    }
};

}  // namespace

void FrontEnd::parseRound(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto round = ngraph::as_type_ptr<ngraph::op::v5::Round>(node);
    IE_ASSERT(node != nullptr);
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Round stage with name {} must have only 1 input, actually provided {} inputs",
                     round->get_name(), inputs.size());

    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Round stage with name {} must have only 1 output, actually provided {} outputs",
                     round->get_name(), outputs.size());

    const auto mode = round->get_mode() == ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO ?
                                           RoundMode::HALF_AWAY_FROM_ZERO : RoundMode::HALF_TO_EVEN;
    auto stage = model->addNewStage<RoundStage>(round->get_name(), StageType::Round, round, inputs, outputs);
    stage->attrs().set("mode", mode);
}

}  // namespace vpu
