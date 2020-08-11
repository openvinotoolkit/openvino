// Copyright (C) 2020 Intel Corporation
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

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

void FrontEnd::parseSwish(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS((inputs.size() == 1) || ((inputs.size() == 2)),
                     "Swish stage with name %s must have 1 or 2 inputs, "
                     "actually provided %d", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Swish stage with name %s must have only 1 output, "
                     "actually provided %d", layer->name, outputs.size());

    model->addNewStage<SwishStage>(layer->name, StageType::Swish, layer, inputs, outputs);
}

}  // namespace vpu
