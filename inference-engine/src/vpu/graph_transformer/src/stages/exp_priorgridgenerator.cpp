// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vpu/frontend/frontend.hpp>

#include <memory>

namespace vpu {

namespace {

VPU_PACKED(ExpPriorGridGeneratorParams {
    int32_t flatten;
    int32_t grid_w;
    int32_t grid_h;
    float   stride_w;
    float   stride_h;
};)

class ExpPriorGridGeneratorStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ExpPriorGridGeneratorStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        for (const auto& inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
             {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
             {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& params = attrs().get<ExpPriorGridGeneratorParams>("params");

        serializer.append(params);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        for (auto& inputEdge : inputEdges()) {
            inputEdge->input()->serializeBuffer(serializer);
        }

        outputEdges()[0]->output()->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseExpPriorGridGenerator(
        const Model& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() == 1);

    ExpPriorGridGeneratorParams params;

    params.flatten  = layer->GetParamAsInt("flatten", 1);
    params.grid_w   = layer->GetParamAsInt("w", 0);
    params.grid_h   = layer->GetParamAsInt("h", 0);
    params.stride_w = layer->GetParamAsFloat("stride_x", 0.0f);
    params.stride_h = layer->GetParamAsFloat("stride_y", 0.0f);

    auto inputPriors     = inputs[0];   // [n][4]
    auto inputFeatureMap = inputs[1];   // [b, c, h, w]
    auto inputImage      = inputs[2];   // [b, 3, im_h, im_w]
    auto outputPriorGrid = outputs[0];  // [m][4]

    const int batch      = inputImage->desc().dim(Dim::N);

    IE_ASSERT((inputPriors->desc().dims().size() == 2) &&
              (inputPriors->desc().dim(Dim::C) == 4));
    IE_ASSERT((inputFeatureMap->desc().dims().size() == 4) &&
              (inputFeatureMap->desc().dim(Dim::N) == batch));
    IE_ASSERT((inputImage->desc().dims().size() == 4) &&
              (inputImage->desc().dim(Dim::C) == 3));

    IE_ASSERT((outputPriorGrid->desc().dims().size() == 2) &&
              (outputPriorGrid->desc().dim(Dim::C) == 4));

    auto stage = model->addNewStage<ExpPriorGridGeneratorStage>(
        layer->name,
        StageType::ExpPriorGridGenerator,
        layer,
        inputs,
        outputs);

    stage->attrs().set("params", params);
}

}  // namespace vpu
