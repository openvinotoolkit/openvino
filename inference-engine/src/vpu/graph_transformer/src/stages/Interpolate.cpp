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

using namespace vpu;
using namespace InferenceEngine;

enum class InterpolateMode {
    nearest,
    linear,
    linear_onnx,
    cubic
};

enum class InterpolateShapeCalcMode {
    sizes,
    scales
};

enum class InterpolateCoordTransMode {
    half_pixel,
    pytorch_half_pixel,
    asymmetric,
    tf_half_pixel_for_nn,
    align_corners
};

enum class InterpolateNearestMode {
    round_prefer_floor,
    round_prefer_ceil,
    floor,
    ceil,
    simple
};

class InterpolateStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<InterpolateStage>(*this);
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
        auto factor = attrs().get<float>("factor");
        auto antialias = attrs().get<bool>("antialias");
        auto axis = attrs().get<std::vector<int>>("InterpolateAxis");
        auto scales = attrs().get<std::vector<float>>("InterpolateScales");
        auto sampleType = attrs().get<InterpolateMode>("type");
        auto sampleNearestMode = attrs().get<InterpolateNearestMode>("nearestMode");
        auto sampleShapeCalcMode = attrs().get<InterpolateShapeCalcMode>("shapeCalcMode");
        auto sampleCoordTransMode = attrs().get<InterpolateCoordTransMode>("coordTransMode");

        serializer.append(static_cast<int32_t>(antialias));
        serializer.append(static_cast<float>(factor));
        serializer.append(static_cast<uint32_t>(sampleType));
        serializer.append(static_cast<uint32_t>(sampleNearestMode));
        serializer.append(static_cast<uint32_t>(sampleShapeCalcMode));
        serializer.append(static_cast<uint32_t>(sampleCoordTransMode));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }

    InterpolateMode mode = InterpolateMode::nearest;
    InterpolateShapeCalcMode shape_calc_mode = InterpolateShapeCalcMode::sizes;
    InterpolateCoordTransMode coordTransMode = InterpolateCoordTransMode::half_pixel;
    InterpolateNearestMode nearestMode       = InterpolateNearestMode::round_prefer_floor;

    bool antialias  = false;
    bool hasPad     = false;
    bool hasSpecifiedAxis = false;
    float cubeCoeff = -0.75;

    std::vector<int> padBegin;
    std::vector<int> padEnd;
    std::vector<int> axes = {0, 1, 2, 3};
    std::vector<float> scales = {1.f, 1.f, 2.f, 2.f};

    SizeVector dstDim;
    SizeVector srcDim;
    SizeVector srcPad;

    InferenceEngine::Precision inputPrecision, outputPrecision;
    size_t srcDataSize, dstDataSize;
};

Stage StageBuilder::addInterpolateStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const std::string& origin) {
    std::cout<<"\naddInterpolateStage\n";
    Stage interpolateStage = model->addNewStage<InterpolateStage>(
        name,
        StageType::Interpolate,
        layer,
        {input},
        {output});
    interpolateStage->attrs().set<std::string>("origin", origin);

    return interpolateStage;
}

void FrontEnd::parseInterpolate(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    ie::details::CaselessEq<std::string> cmp;

    _stageBuilder->addInterpolateStage(model, layer->name, layer, inputs[0], outputs[0], "parseInterpolate");

    auto stage = model->addNewStage<InterpolateStage>(layer->name, StageType::Interpolate, layer, inputs, outputs);

    stage->attrs().set<bool>("antialias", layer->GetParamAsInt("antialias", 0));
    stage->attrs().set<float>("factor", layer->GetParamAsInt("factor", -1.0f));

    auto method = layer->GetParamAsString("type", "nearest");
    if (cmp(method, "nearest")) {
        stage->attrs().set<InterpolateMode>("type", InterpolateMode::nearest);
    } else if (cmp(method, "linear")) {
        stage->attrs().set<InterpolateMode>("type", InterpolateMode::linear);
    } else if (cmp(method, "cubic")) {
        stage->attrs().set<InterpolateMode>("type", InterpolateMode::cubic);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this Interpolate type";
    }

    auto nnMode = layer->GetParamAsString("nearestMode", "round_prefer_floor");
    if (cmp(nnMode, "round_prefer_floor")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::round_prefer_floor);
    } else if (cmp(nnMode, "round_prefer_ceil")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::round_prefer_ceil);
    } else if (cmp(nnMode, "floor")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::floor);
    } else if (cmp(nnMode, "ceil")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::ceil);
    } else if (cmp(nnMode, "simple")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::simple);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this Interpolate nearest mode variant";
    }

    auto shapeCalcMode = layer->GetParamAsString("shapeCalcMode", "sizes");
    if (cmp(shapeCalcMode, "sizes")) {
        stage->attrs().set<InterpolateShapeCalcMode>("shapeCalcMode", InterpolateShapeCalcMode::sizes);
    } else if (cmp(shapeCalcMode, "scales")) {
        stage->attrs().set<InterpolateShapeCalcMode>("shapeCalcMode", InterpolateShapeCalcMode::scales);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this Interpolate shape calculation mode";
    }

    auto coordTransMode = layer->GetParamAsString("coordTransMode", "half_pixel");
    if (cmp(coordTransMode, "half_pixel")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::half_pixel);
    } else if (cmp(coordTransMode, "pytorch_half_pixel")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::pytorch_half_pixel);
    } else if (cmp(coordTransMode, "pytorch_half_pixel")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::pytorch_half_pixel);
    } else if (cmp(coordTransMode, "tf_half_pixel_for_nn")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::tf_half_pixel_for_nn);
    } else if (cmp(coordTransMode, "align_corners")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::align_corners);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this Interpolate coordinate transform mode";
    }
}
