// Copyright (C) 2018-2020 Intel Corporation
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

class ScaleStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ScaleStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

Stage StageBuilder::addScaleStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& scales,
        const Data& output) {
    return model->addNewStage<ScaleStage>(
        name,
        StageType::Scale,
        layer,
        {input, scales},
        {output});
}

void FrontEnd::parseScale(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::ScaleShiftLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    if (layer->_broadcast != 0) {
        VPU_THROW_EXCEPTION <<
            "Layer " << layer->name << " doesn't support broadcast param";
    }

    auto input = inputs[0];
    auto output = outputs[0];

    Data scales, biases;
    std::tie(scales, biases) = getWeightsAndBiases(model, layer);

    if (biases->usage() == DataUsage::Fake) {
        model->addNewStage<ScaleStage>(
            layer->name,
            StageType::Scale,
            layer,
            {input, scales},
            {output});
    } else {
        model->addNewStage<ScaleStage>(
            layer->name,
            StageType::ScaleShift,
            layer,
            {input, scales, biases},
            {output});
    }
}

}  // namespace vpu
