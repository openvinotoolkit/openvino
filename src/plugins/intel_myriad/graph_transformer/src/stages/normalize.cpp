// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <vector>
#include <map>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class NormalizeStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<NormalizeStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        if (input(0)->desc().dimsOrder().dimInd(Dim::C) == 0) {
            stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
            stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto acrossSpatial = attrs().get<bool>("acrossSpatial");
        auto channelShared = attrs().get<bool>("channelShared");
        auto eps = attrs().get<float>("eps");

        serializer.append(static_cast<int32_t>(acrossSpatial));
        serializer.append(static_cast<int32_t>(channelShared));
        serializer.append(static_cast<float>(eps));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto scales = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        scales->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseNormalize(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto acrossSpatial = layer->GetParamAsInt("across_spatial", 0);
    auto channelShared = layer->GetParamAsInt("channel_shared", 0);
    float eps = layer->GetParamAsFloat("eps", 0.0f);

    auto weightsIt = layer->blobs.find("weights");
    if (weightsIt == layer->blobs.end()) {
        VPU_THROW_EXCEPTION << "Missing weights for " << layer->name << " layer";
    }

    auto weightsBlob = weightsIt->second;
    IE_ASSERT(weightsBlob != nullptr);

    auto output = outputs[0];

    auto scales = model->addConstData(layer->name + "@scales", DataDesc({weightsBlob->size()}), ieBlobContent(weightsBlob));

    auto stage = model->addNewStage<NormalizeStage>(layer->name, StageType::Normalize, layer, {inputs[0], scales}, outputs);
    stage->attrs().set<bool>("acrossSpatial", acrossSpatial);
    stage->attrs().set<bool>("channelShared", channelShared);
    stage->attrs().set<float>("eps", eps);
}

}  // namespace vpu
