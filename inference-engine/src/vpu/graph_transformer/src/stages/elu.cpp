// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>

#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class EluStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<EluStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto alpha = attrs().get<float>("alpha");

        serializer.append(static_cast<float>(alpha));
    }
};

}  // namespace

void FrontEnd::parseELU(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const auto& elu = ngraph::as_type_ptr<ngraph::op::Elu>(node);
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto alpha = elu->get_alpha();

    auto stage = model->addNewStage<EluStage>(elu->get_name(), StageType::Elu, elu, inputs, outputs);
    stage->attrs().set<float>("alpha", alpha);
}

}  // namespace vpu
