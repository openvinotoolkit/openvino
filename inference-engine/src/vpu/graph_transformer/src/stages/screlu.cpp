// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <cmath>

#include <vector>
#include <limits>
#include <memory>
#include <set>
#include <string>

namespace vpu {

namespace {
class SCReluStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<SCReluStage>(*this);
    }
    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto negativeSlope = attrs().get<float>("negativeSlope");

        serializer.append(negativeSlope);
        auto axis = attrs().get<Dim>("axis");
        auto input = inputEdge(0)->input();
        auto axisInd = input->desc().dimsOrder().dimInd(axis);

        serializer.append(static_cast<int32_t>(axisInd));
    }
    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        IE_ASSERT(inputEdges().size() == 3);
        IE_ASSERT(outputEdges().size() == 1);

        auto input = inputEdge(0)->input();
        auto inputScales = inputEdge(1)->input();
        auto inputBiases = inputEdge(2)->input();
        auto output = outputEdge(0) ->output();

        auto inDesc = input->desc();
        auto inScaleDesc = inputScales->desc();
        auto inBiasDesc = inputBiases->desc();

        auto finalOrder = inDesc.dimsOrder();

        IE_ASSERT(inBiasDesc.numDims() == 1);
        IE_ASSERT(inScaleDesc.numDims() == 1);

        auto biasesOrder = inBiasDesc.dimsOrder();
        auto scalesOrder = inScaleDesc.dimsOrder();

        orderInfo.setInput(inputEdge(0), finalOrder);
        orderInfo.setInput(inputEdge(1), scalesOrder);
        orderInfo.setInput(inputEdge(2), biasesOrder);
        orderInfo.setOutput(outputEdge(0), finalOrder);
    }
    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }
    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }
    void finalizeDataLayoutImpl() override {
    }
    void finalCheckImpl() const override {
    }
    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 1 || numInputs() == 2 || numInputs() == 3);
        IE_ASSERT(numOutputs() == 1);

        assertAllInputsOutputsTypes(this, DataType::FP16, DataType::FP16);
    }
    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(inputEdges().size() == 3);
        IE_ASSERT(outputEdges().size() == 1);

        auto input = inputEdge(0)->input();
        auto inputScales = inputEdge(1)->input();
        auto inputBiases = inputEdge(2)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer, output->desc().dimsOrder());
        output->serializeBuffer(serializer);
        inputScales->serializeBuffer(serializer, output->desc().dimsOrder());
        inputBiases->serializeBuffer(serializer, output->desc().dimsOrder());
    }
};

}  // namespace

Stage StageBuilder::addSCReluStage(
        const vpu::Model &model,
        const std::string &name,
        const InferenceEngine::CNNLayerPtr &layer,
        float negativeSlope,
        Dim axis,
        const vpu::Data &input,
        const vpu::Data &output,
        const vpu::Data &scales,
        const vpu::Data &biases
        ) {
    auto stageType = StageType::SCRelu;
    const Data& fakeInput = model->addFakeData();
    auto stage = model->addNewStage<SCReluStage>(
            name,
            stageType,
            layer,
            {input},
            {output});

    if (scales == nullptr) {
        model->addStageInput(stage, fakeInput);
    } else {
        model->addStageInput(stage, scales);
    }

    if (biases == nullptr) {
        model->addStageInput(stage, fakeInput);
    } else {
        model->addStageInput(stage, biases);
    }

    stage->attrs().set<float>("negativeSlope", negativeSlope);
    stage->attrs().set("axis", axis);

    return stage;
}

}  // namespace vpu
