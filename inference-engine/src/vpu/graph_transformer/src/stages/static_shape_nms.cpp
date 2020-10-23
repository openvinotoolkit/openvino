// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <precision_utils.h>

#include <memory>
#include <set>

namespace vpu {

namespace {

class StaticShapeNMS final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<StaticShapeNMS>(*this);
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
        // Current NMS implementation doesn't allow calculation of `> boxesThreshold` boxes using one SHAVE
        constexpr int boxesThreshold = 3650;

        const auto& inDesc = input(0)->desc();
        const auto& maxBoxesNum = inDesc.dim(Dim::H);

        return maxBoxesNum <= boxesThreshold ? StageSHAVEsRequirements::OnlyOne : StageSHAVEsRequirements::NeedMax;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
                                 {{DataType::FP16},
                                  {DataType::FP16},
                                  {DataType::S32},
                                  {DataType::FP16},
                                  {DataType::FP16}},
                                 {{DataType::S32},
                                  {DataType::S32}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto center_point_box = attrs().get<bool>("center_point_box");

        serializer.append(static_cast<int32_t>(center_point_box));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input1 = inputEdges()[0]->input();
        auto input2 = inputEdges()[1]->input();
        auto input3 = inputEdges()[2]->input();
        auto input4 = inputEdges()[3]->input();
        auto input5 = inputEdges()[4]->input();
        auto outputData = outputEdges()[0]->output();
        auto outputDims = outputEdges()[1]->output();

        input1->serializeBuffer(serializer);
        input2->serializeBuffer(serializer);
        input3->serializeBuffer(serializer);
        input4->serializeBuffer(serializer);
        input5->serializeBuffer(serializer);
        outputData->serializeBuffer(serializer);
        outputDims->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseStaticShapeNMS(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 6,
        "StaticShapeNMS with name {} parsing failed, expected number of inputs: 6, but {} provided",
        layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 3,
        "StaticShapeNMS with name {} parsing failed, expected number of outputs: 4, but {} provided",
        layer->name, outputs.size());

    const auto softNMSSigmaData = inputs[5];
    VPU_THROW_UNLESS(softNMSSigmaData->usage() == DataUsage::Const,
        "StaticShapeNMS with name {} parsing failed: softNMSSigma should have usage {} while it actually has {}",
        layer->type, DataUsage::Const, softNMSSigmaData->usage());
    VPU_THROW_UNLESS(softNMSSigmaData->desc().totalDimSize() == 1,
        "StaticShapeNMS with name {} parsing failed: softNMSSigma input should contain 1 value, while it has {} values",
        layer->type, softNMSSigmaData->desc().totalDimSize());
    const auto softNMSSigma = InferenceEngine::PrecisionUtils::f16tof32(softNMSSigmaData->content()->get<InferenceEngine::ie_fp16>()[0]);
    VPU_THROW_UNLESS(softNMSSigma == 0,
        "StaticShapeNMS with name {} parsing failed: the only supported value for softNMSSigma is 0, while it actually equal to {}",
        layer->name, softNMSSigma);

    auto usedInputs = inputs;
    // Erase unused softNMSSigma input
    usedInputs.pop_back();

    const auto& outIndices = outputs[0];
    const auto& outScores = outputs[1];
    const auto& outShape = outputs[2];

    VPU_THROW_UNLESS(outScores == nullptr,
        "StaticShapeNMS with name {} parsing failed: selected_scores output is not supported {}",
        layer->name);

    const auto sortResultDescending = layer->GetParamAsBool("sort_result_descending");
    const auto centerPointBox = layer->GetParamAsBool("center_point_box");

    VPU_THROW_UNLESS(sortResultDescending == false,
        "StaticShapeNMS with name {}: parameter sortResultDescending=true is not supported on VPU", layer->name);

    auto stage = model->addNewStage<StaticShapeNMS>(layer->name, StageType::StaticShapeNMS, layer, usedInputs, DataVector{outIndices, outShape});
    stage->attrs().set<bool>("center_point_box", centerPointBox);
}

}  // namespace vpu
