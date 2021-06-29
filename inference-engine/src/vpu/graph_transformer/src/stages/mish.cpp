// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class MishStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<MishStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NeedMax;
    }
};

}  // namespace

void FrontEnd::parseMish(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Mish stage with name %s must have only 1 input, "
                     "actually provided %d", node->get_name(), inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Mish stage with name %s must have only 1 output, "
                     "actually provided %d", node->get_name(), outputs.size());

    model->addNewStage<MishStage>(node->get_name(), StageType::Mish, node, inputs, outputs);
}

}  // namespace vpu
