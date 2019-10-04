// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <string>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

void FrontEnd::parseBias(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto biases = inputs[1];
    auto output = outputs[0];

    _stageBuilder->addBiasStage(
        model,
        layer->name,
        layer,
        input, biases,
        output);
}

namespace {
class BiasStage final : public PostOpStage {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<BiasStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step,
            StageDataInfo<float>& scaleInfo) override {
        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales[0];

            scaleInfo.setInput(inputEdge(1), inputScale);
            scaleInfo.setOutput(outputEdge(0), inputScale);
        } else {
            // Bias can only propagate scaling, not generate.
            scaleInfo.setInput(inputEdge(0), 1.0f);
            scaleInfo.setInput(inputEdge(1), 1.0f);
            scaleInfo.setOutput(outputEdge(0), 1.0f);
        }
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

Stage StageBuilder::addBiasStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& biases,
        const Data& output) {
    return model->addNewStage<BiasStage>(
        name,
        StageType::Bias,
        layer,
        {input, biases},
        {output});
}

}  // namespace vpu
