// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class GeluStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<GeluStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
    }
};

}  // namespace

void FrontEnd::parseGelu(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Gelu stage with name %s must have only 1 input, "
                     "actually provided %d", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Gelu stage with name %s must have only 1 output, "
                     "actually provided %d", layer->name, outputs.size());

    model->addNewStage<GeluStage>(layer->name, StageType::Gelu, layer, inputs, outputs);
}

}  // namespace vpu
