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

class CeilingStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<CeilingStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

void FrontEnd::parseCeiling(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Ceiling stage with name {} must have only 1 input, actually provided {} inputs",
                     layer->name, inputs.size());

    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Ceiling stage with name {} must have only 1 output, actually provided {} outputs",
                     layer->name, outputs.size());

    model->addNewStage<CeilingStage>(layer->name, StageType::Ceiling, layer, inputs, outputs);
}

}  // namespace vpu
