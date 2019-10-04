// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>
#include <precision_utils.h>

namespace vpu {

namespace {

class MVNStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<MVNStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto normalize = attrs().get<int>("normalize");
        auto across_channels = attrs().get<int>("across_channels");
        auto eps = attrs().get<float>("eps");

        serializer.append(static_cast<int32_t>(normalize));
        serializer.append(static_cast<int32_t>(across_channels));
        serializer.append(static_cast<float>(eps));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeOldBuffer(handle_from_this(), serializer);
        output->serializeOldBuffer(handle_from_this(), serializer);
    }
};

}  // namespace

void FrontEnd::parseMVN(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::MVNLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto stage = model->addNewStage<MVNStage>(
        layer->name,
        StageType::MVN,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("normalize", layer->normalize);
    stage->attrs().set<int>("across_channels", layer->across_channels);
    stage->attrs().set<float>("eps", layer->GetParamAsFloat("eps", 0.0f));
}

}  // namespace vpu
