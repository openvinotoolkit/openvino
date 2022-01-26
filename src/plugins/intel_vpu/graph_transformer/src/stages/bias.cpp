// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <string>

#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

void FrontEnd::parseBias(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto biases = inputs[1];
    auto output = outputs[0];

    _stageBuilder->addBiasStage(model, layer->name, layer, input, biases, output);
}

namespace {
class BiasStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<BiasStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

Stage StageBuilder::addBiasStage(
        const Model& model,
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
