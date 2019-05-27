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

namespace {

class ClampStage final : public PostOpStage {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ClampStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        DataMap<float> out;

        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales.at(input);

            out[output] = inputScale;

            attrs().get<float>("min_value") *= inputScale;
            attrs().get<float>("max_value") *= inputScale;
        } else {
            // Clamp can only propagate scaling, not generate.
            out[input] = 1.0f;
            out[output] = 1.0f;
        }

        return out;
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto min_value = attrs().get<float>("min_value");
        auto max_value = attrs().get<float>("max_value");

        serializer.append(static_cast<float>(min_value));
        serializer.append(static_cast<float>(max_value));
    }
};

}  // namespace

void FrontEnd::parseClamp(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::ClampLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    _stageBuilder->addClampStage(model, layer->name, layer, layer->min_value,  layer->max_value, inputs[0], outputs[0]);
}

Stage StageBuilder::addClampStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float min,
            float max,
            const Data& input,
            const Data& output) {
        auto stage = model->addNewStage<ClampStage>(
                name,
                StageType::Clamp,
                layer,
                {input},
                {output});

        stage->attrs().set<float>("min_value", min);
        stage->attrs().set<float>("max_value", max);

        return stage;
    }


}  // namespace vpu
