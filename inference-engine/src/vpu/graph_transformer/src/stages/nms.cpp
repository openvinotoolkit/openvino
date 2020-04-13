// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <memory>
#include <set>

namespace vpu {

namespace {

class NonMaxSuppression final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<NonMaxSuppression>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::OnlyOne;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
                                 {{DataType::FP16},
                                  {DataType::FP16},
                                  {DataType::S32},
                                  {DataType::FP16},
                                  {DataType::FP16}},
                                 {{DataType::S32}});
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        bool center_point_box = attrs().get<bool>("center_point_box");

        serializer.append(static_cast<int32_t>(center_point_box));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(inputEdges().size() >= 2 && inputEdges().size() <= 5);
        IE_ASSERT(outputEdges().size() == 1);

        auto input1 = inputEdges()[0]->input();
        auto input2 = inputEdges()[1]->input();
        auto input3 = inputEdges()[2]->input();
        auto input4 = inputEdges()[3]->input();
        auto input5 = inputEdges()[4]->input();
        auto output = outputEdges()[0]->output();

        input1->serializeBuffer(serializer);
        input2->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        input3->serializeBuffer(serializer);
        input4->serializeBuffer(serializer);
        input5->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseNonMaxSuppression(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    auto layer = std::dynamic_pointer_cast<ie::NonMaxSuppressionLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    IE_ASSERT(inputs.size() >= 2 && inputs.size() <= 5);
    IE_ASSERT(outputs.size() == 1);

    DataVector tempInputs = inputs;
    for (size_t fake = inputs.size(); fake < 5; fake++) {
        tempInputs.push_back(model->addFakeData());
    }

    auto stage = model->addNewStage<NonMaxSuppression>(layer->name, StageType::NonMaxSuppression, layer, tempInputs, outputs);
    stage->attrs().set<bool>("center_point_box", layer->center_point_box);
}

}  // namespace vpu
