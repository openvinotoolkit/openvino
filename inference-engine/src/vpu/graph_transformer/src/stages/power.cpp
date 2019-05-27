// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>
#include <string>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

void FrontEnd::parsePower(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    auto layer = std::dynamic_pointer_cast<ie::PowerLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    _stageBuilder->addPowerStage(
        model,
        layer->name,
        layer,
        layer->scale,
        layer->power,
        layer->offset,
        inputs[0],
        outputs[0]);
}

namespace {

class PowerStage final : public PostOpStage {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PowerStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto power = attrs().get<float>("power");
        auto& scale = attrs().get<float>("scale");
        auto& bias = attrs().get<float>("bias");

        DataMap<float> out;

        if (power != 1.0f) {
            out[input] = 1.0f;
            out[output] = 1.0f;
        } else {
            auto inputScale = inputScales.at(input);

            out[output] = inputScale;

            if (step == ScalePropagationStep::ScaleInput) {
                scale *= inputScale;
            }
            if (step != ScalePropagationStep::Check) {
                bias *= inputScale;
            }
        }

        return out;
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
        const Model::Ptr& model,
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
