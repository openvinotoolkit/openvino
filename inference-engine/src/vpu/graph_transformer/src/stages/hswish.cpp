// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class HSwishStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<HSwishStage >(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

void FrontEnd::parseHSwish(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS((inputs.size() == 1),
                     "HSwish stage with name {} must have only 1 input, "
                     "actually provided {}", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "HSwish stage with name {} must have only 1 output, "
                     "actually provided {}", layer->name, outputs.size());

    model->addNewStage<HSwishStage>(layer->name, StageType::HSwish, layer, inputs, outputs);
}

}  // namespace vpu
