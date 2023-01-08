// Copyright (C) 2018-2022 Intel Corporation
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

void FrontEnd::parseRound(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Round stage with name {} must have only 1 input, actually provided {} inputs",
                     layer->name, inputs.size());

    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Round stage with name {} must have only 1 output, actually provided {} outputs",
                     layer->name, outputs.size());

    const std::map<std::string, RoundMode> modeFromString = {
        {"half_to_even", RoundMode::HALF_TO_EVEN},
        {"half_away_from_zero", RoundMode::HALF_AWAY_FROM_ZERO}
    };

    const auto modeString = layer->GetParamAsString("mode", "half_to_even");
    const auto& modeFind = modeFromString.find(modeString);
    VPU_THROW_UNLESS(modeFind != modeFromString.end(),
                    "{} layer with name {}: Graph Transformer doesn't support {} mode",
                    layer->type, layer->name, modeString);

    const auto mode = modeFind->second;
    auto stage = model->addNewStage<RoundStage>(layer->name, StageType::Round, layer, inputs, outputs);
    stage->attrs().set("mode", mode);
}

}  // namespace vpu
