// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/stages/post_op_stage.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/model/data_contents/prelu_blob_content.hpp>

#include <vector>
#include <memory>

namespace vpu {

namespace {

class PReluStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PReluStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

void FrontEnd::parsePReLU(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    
    const auto& prelu = ngraph::as_type_ptr<ngraph::opset4::PRelu>(node);
    VPU_THROW_UNLESS(prelu != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    // IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);
    auto inputNode = node->input_value(1).get_node_shared_ptr();
    auto weightsBlob = shareWeights(inputNode);

    IE_ASSERT(weightsBlob != nullptr);

    auto channelShared =  0;

    auto output = outputs[0];

    auto weights = model->addConstData(
        prelu->get_friendly_name() + "@weights",
        DataDesc({output->desc().dim(Dim::C)}),
        std::make_shared<PReLUBlobContent>(weightsBlob, DataDesc({output->desc().dim(Dim::C)}),
                                           channelShared ? output->desc().dim(Dim::C) : 1));

    model->addNewStage<PReluStage>(prelu->get_friendly_name(), StageType::PRelu, prelu, {inputs[0], weights}, outputs);
}

}  // namespace vpu
