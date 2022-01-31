// Copyright (C) 2018-2022 Intel Corporation
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

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto beta = attrs().get<float>("beta");

        serializer.append(static_cast<float>(beta));
    }
};

}  // namespace

void FrontEnd::parseSwish(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS((inputs.size() == 1),
                     "Swish stage with name %s must have 1 input, "
                     "actually provided %d", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Swish stage with name %s must have only 1 output, "
                     "actually provided %d", layer->name, outputs.size());

    auto stage = model->addNewStage<SwishStage>(
        layer->name, StageType::Swish, layer, inputs, outputs);

    stage->attrs().set<float>("beta", layer->GetParamAsFloat("alpha"));
}

}  // namespace vpu
