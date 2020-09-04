// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stages/nms.hpp>

#include <vpu/frontend/frontend.hpp>
#include <memory>
#include <set>

namespace vpu {

StagePtr NonMaxSuppression::cloneImpl() const {
    return std::make_shared<NonMaxSuppression>(*this);
}

void NonMaxSuppression::propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) {
}

void NonMaxSuppression::getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) {
}

void NonMaxSuppression::finalizeDataLayoutImpl() {
}

void NonMaxSuppression::getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) {
}

StageSHAVEsRequirements NonMaxSuppression::getSHAVEsRequirementsImpl() const {
    // Current NMS implementation doesn't allow calculation of `> boxesThreshold` boxes using one SHAVE
    constexpr int boxesThreshold = 3650;

    const auto& inDesc = input(0)->desc();
    const auto& maxBoxesNum = inDesc.dim(Dim::H);

    if (maxBoxesNum > boxesThreshold) {
        return StageSHAVEsRequirements::NeedMax;
    } else {
        return StageSHAVEsRequirements::OnlyOne;
    }
}

void NonMaxSuppression::initialCheckImpl() const {
    assertInputsOutputsTypes(this,
                             {{DataType::FP16},
                              {DataType::FP16},
                              {DataType::S32},
                              {DataType::FP16},
                              {DataType::FP16}},
                             {{DataType::S32}});
}

void NonMaxSuppression::finalCheckImpl() const {
}

void NonMaxSuppression::serializeParamsImpl(BlobSerializer& serializer) const {
    bool center_point_box = attrs().get<bool>("center_point_box");

    serializer.append(static_cast<int32_t>(center_point_box));
}

void NonMaxSuppression::serializeDataImpl(BlobSerializer& serializer) const {
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
