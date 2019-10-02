// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class RegionYoloStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<RegionYoloStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        if (attrs().get<bool>("doSoftMax")) {
            // Major dimension must be compact.
            stridesInfo.setInput(inputEdge(0), StridesRequirement().add(2, DimStride::Compact));
        }
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
        auto classes = attrs().get<int>("classes");
        auto coords = attrs().get<int>("coords");
        auto num = attrs().get<int>("num");
        auto maskSize = attrs().get<int>("maskSize");
        auto doSoftMax = attrs().get<bool>("doSoftMax");

        serializer.append(static_cast<int32_t>(classes));
        serializer.append(static_cast<int32_t>(coords));
        serializer.append(static_cast<int32_t>(num));
        serializer.append(static_cast<int32_t>(maskSize));
        serializer.append(static_cast<int32_t>(doSoftMax));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeOldBuffer(this, serializer);
        output->serializeOldBuffer(this, serializer);
    }
};

}  // namespace

void FrontEnd::parseRegionYolo(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto mask = layer->GetParamAsInts("mask", {});

    auto stage = model->addNewStage<RegionYoloStage>(
        layer->name,
        StageType::RegionYolo,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("classes", layer->GetParamAsInt("classes", 20));
    stage->attrs().set<int>("coords", layer->GetParamAsInt("coords", 4));
    stage->attrs().set<int>("num", layer->GetParamAsInt("num", 5));
    stage->attrs().set<int>("maskSize", static_cast<int>(mask.size()));
    stage->attrs().set<bool>("doSoftMax", layer->GetParamAsInt("do_softmax", 1));
}

}  // namespace vpu
