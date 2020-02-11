// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>

#include <vpu/stages/post_op_stage.hpp>

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

void FrontEnd::parsePReLU(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto weightsIt = layer->blobs.find("weights");
    if (weightsIt == layer->blobs.end()) {
        THROW_IE_EXCEPTION << "[VPU] PReLU doesn't have weights";
    }

    auto weightsBlob = weightsIt->second;
    IE_ASSERT(weightsBlob != nullptr);

    auto channelShared = layer->GetParamAsInt("channel_shared", 0);

    auto output = outputs[0];

    auto weights = model->addConstData(
        layer->name + "@weights",
        DataDesc({output->desc().dim(Dim::C)}),
        ieBlobContent(weightsBlob, channelShared ? output->desc().dim(Dim::C) : 1));

    model->addNewStage<PReluStage>(layer->name, StageType::PRelu, layer, {inputs[0], weights}, outputs);
}

}  // namespace vpu
