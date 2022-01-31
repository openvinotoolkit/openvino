// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>
#include <string>

#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

void FrontEnd::parsePower(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    auto layer = std::dynamic_pointer_cast<ie::PowerLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    _stageBuilder->addPowerStage(model, layer->name, layer, layer->scale, layer->power, layer->offset, inputs[0], outputs[0]);
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
        const ie::CNNLayerPtr& layer,
        float scale,
        float power,
        float bias,
        const Data& input,
        const Data& output) {
    auto stage = model->addNewStage<PowerStage>(
        name,
        StageType::Power,
        layer,
        {input},
        {output});

    stage->attrs().set<float>("scale", scale);
    stage->attrs().set<float>("power", power);
    stage->attrs().set<float>("bias", bias);

    return stage;
}

}  // namespace vpu
