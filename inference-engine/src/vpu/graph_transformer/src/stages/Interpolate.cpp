// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <ie_common.h>
#include <ie_blob.h>
#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

using namespace InferenceEngine;

namespace vpu {
enum class InterpolateMode {
    nearest     = 0,
    linear      = 1,
    linear_onnx = 2,
    cubic       = 3
};

enum class InterpolateShapeCalcMode {
    sizes  = 0,
    scales = 1
};

enum class InterpolateCoordTransMode {
    half_pixel           = 0,
    pytorch_half_pixel   = 1,
    asymmetric           = 2,
    tf_half_pixel_for_nn = 3,
    align_corners        = 4
};

enum class InterpolateNearestMode {
    round_prefer_floor = 0,
    round_prefer_ceil  = 1,
    floor              = 2,
    ceil               = 3,
    simple             = 4
};

namespace {
class InterpolateStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<InterpolateStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto input2 = inputEdge(2)->input();
        auto input3 = inputEdge(3)->input();

        orderInfo.setOutput(outputEdge(0), input0->desc().dimsOrder());
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
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::S32}, {DataType::FP16}, {DataType::S32}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto perm = input->desc().dimsOrder().toPermutation();
        IE_ASSERT(perm.size() <= 4);

        auto antialias = attrs().get<bool>("antialias");
        auto cube_coeff = attrs().get<float>("cube_coeff");
        auto batch = attrs().get<int>("batch");
        auto sampleType = attrs().get<int>("type");
        auto sampleNearestMode = attrs().get<int>("nearestMode");
        auto sampleShapeCalcMode = attrs().get<int>("shapeCalcMode");
        auto sampleCoordTransMode = attrs().get<int>("coordTransMode");
        auto pads_begin = attrs().get<DimValues>("pads_begin");
        auto pads_end = attrs().get<DimValues>("pads_end");

        serializer.append(static_cast<bool>(antialias));
        serializer.append(static_cast<float>(cube_coeff));
        serializer.append(static_cast<int>(batch));
        serializer.append(static_cast<int>(sampleType));
        serializer.append(static_cast<int>(sampleNearestMode));
        serializer.append(static_cast<int>(sampleShapeCalcMode));
        serializer.append(static_cast<int>(sampleCoordTransMode));

        for (int i = 0; i < perm.size(); ++i) {
            serializer.append(static_cast<int>(pads_begin.get(perm[i], 0)));
            serializer.append(static_cast<int>(pads_end.get(perm[i], 0)));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto input2 = inputEdge(2)->input();
        auto input3 = inputEdge(3)->input();
        auto output = outputEdge(0)->output();

        input0->serializeBuffer(serializer);
        input1->serializeBuffer(serializer);
        input2->serializeBuffer(serializer);
        input3->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addInterpolateStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const DataVector& input,
        const Data& output,
        const std::string& origin) {
    Stage interpolateStage = model->addNewStage<InterpolateStage>(
        name,
        StageType::Interpolate,
        layer,
        {input[0], input[1], input[2], input[3]},
        {output});
    interpolateStage->attrs().set<std::string>("origin", origin);

    return interpolateStage;
}

void FrontEnd::parseInterpolate(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    printf("PARSE start\n");
    IE_ASSERT(inputs.size() == 4);
    IE_ASSERT(outputs.size() == 1);

    int ON = outputs[0]->desc().dim(Dim::N);
    int OC = outputs[0]->desc().dim(Dim::C);
    int OH = outputs[0]->desc().dim(Dim::H);
    int OW = outputs[0]->desc().dim(Dim::W);

    int IN = inputs[0]->desc().dim(Dim::N);
    int IC = inputs[0]->desc().dim(Dim::C);
    int IH = inputs[0]->desc().dim(Dim::H);
    int IW = inputs[0]->desc().dim(Dim::W);

    printf("ON = %d, OC = %d, OH = %d, OW = %d\n", ON, OC, OH, OW);
    printf("IN = %d, IC = %d, IH = %d, IW = %d\n", IN, IC, IH, IW);

    auto stage = model->addNewStage<InterpolateStage>(layer->name, StageType::Interpolate, layer, inputs, outputs);

    stage->attrs().set<bool>("antialias", layer->GetParamAsBool("antialias", 0));
    stage->attrs().set<float>("cube_coeff", layer->GetParamAsFloat("cube_coeff", 0));
    stage->attrs().set<int>("batch", layer->GetParamAsInt("batch", 1));
    stage->attrs().set<int>("type", layer->GetParamAsInt("type", 0));

    stage->attrs().set<std::vector<int>>("pads_begin", layer->GetParamAsInts("pads_begin"));
    stage->attrs().set<std::vector<int>>("pads_end", layer->GetParamAsInts("pads_end"));

    stage->attrs().set<int>("nearestMode", layer->GetParamAsInt("nearestMode", 0));
    stage->attrs().set<int>("shapeCalcMode", layer->GetParamAsInt("shapeCalcMode", 0));
    stage->attrs().set<int>("coordTransMode", layer->GetParamAsInt("coordTransMode", 0));
    printf("PARSE end\n");
}

}  // namespace vpu
