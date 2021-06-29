// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class SoftPlusStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<SoftPlusStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

void FrontEnd::parseSoftPlus(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "SoftPlus stage with name %s must have only 1 input, "
                     "actually provided %d", node->get_name(), inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "SoftPlus stage with name %s must have only 1 output, "
                     "actually provided %d", node->get_name(), outputs.size());

    model->addNewStage<SoftPlusStage>(node->get_name(), StageType::SoftPlus, node, inputs, outputs);
}

}  // namespace vpu
