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
        auto output = outputEdge(0)->output();

        orderInfo.setOutput(outputEdge(0), input0->desc().dimsOrder());
        // orderInfo.setInput(inputEdge(0), DimsOrder::fromNumDims(input0->desc().numDims()));
        // orderInfo.setOutput(outputEdge(0), DimsOrder::fromNumDims(output->desc().numDims()));
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
        auto sampleType = attrs().get<InterpolateMode>("type");
        auto sampleNearestMode = attrs().get<InterpolateNearestMode>("nearestMode");
        auto sampleShapeCalcMode = attrs().get<InterpolateShapeCalcMode>("shapeCalcMode");
        auto sampleCoordTransMode = attrs().get<InterpolateCoordTransMode>("coordTransMode");
        auto pads_begin = attrs().get<std::vector<int>>("pads_begin");
        auto pads_end = attrs().get<std::vector<int>>("pads_end");

        serializer.append(static_cast<bool>(antialias));
        serializer.append(static_cast<float>(cube_coeff));
        serializer.append(static_cast<uint32_t>(sampleType));
        serializer.append(static_cast<uint32_t>(sampleNearestMode));
        serializer.append(static_cast<uint32_t>(sampleShapeCalcMode));
        serializer.append(static_cast<uint32_t>(sampleCoordTransMode));
        serializer.append(static_cast<std::vector<int>>(pads_begin));
        serializer.append(static_cast<std::vector<int>>(pads_end));
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

    ie::details::CaselessEq<std::string> cmp;

    auto stage = model->addNewStage<InterpolateStage>(layer->name, StageType::Interpolate, layer, inputs, outputs);

    stage->attrs().set<bool>("antialias", layer->GetParamAsBool("antialias", 0));
    stage->attrs().set<float>("cube_coeff", layer->GetParamAsFloat("cube_coeff", 0));

    auto mode = layer->GetParamAsString("type", "nearest");
    auto nearest_m = layer->GetParamAsString("nearestMode", "round_prefer_floor");
    auto shape_calc = layer->GetParamAsString("shapeCalcMode", "scales");
    auto coord_trans = layer->GetParamAsString("coordTransMode", "half_pixel");

    if (cmp(mode, "nearest")) {
        stage->attrs().set<InterpolateMode>("type", InterpolateMode::nearest);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " supports only nearest mode";
    }

    if (cmp(nearest_m, "round_prefer_floor")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::round_prefer_floor);
    } else if (cmp(nearest_m, "round_prefer_ceil")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::round_prefer_ceil);
    } else if (cmp(nearest_m, "floor")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::floor);
    } else if (cmp(nearest_m, "ceil")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::ceil);
    } else if (cmp(nearest_m, "simple")) {
        stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::simple);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " does not support this nearest mode";
    }

    if (cmp(shape_calc, "scales")) {
        stage->attrs().set<InterpolateShapeCalcMode>("shapeCalcMode", InterpolateShapeCalcMode::scales);
    } else if (cmp(shape_calc, "sizes")) {
        stage->attrs().set<InterpolateShapeCalcMode>("shapeCalcMode", InterpolateShapeCalcMode::sizes);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " does not support this shape calculation mode";
    }

    if (cmp(coord_trans, "half_pixel")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::half_pixel);
    } else if (cmp(coord_trans, "pytorch_half_pixel")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::pytorch_half_pixel);
    } else if (cmp(coord_trans, "asymmetric")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::asymmetric);
    } else if (cmp(coord_trans, "tf_half_pixel_for_nn")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::tf_half_pixel_for_nn);
    } else if (cmp(coord_trans, "align_corners")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::align_corners);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " does not support this coordinate transformation mode";
    }
  
    stage->attrs().set<std::vector<int>>("pads_begin", layer->GetParamAsInts("pads_begin"));
    stage->attrs().set<std::vector<int>>("pads_end", layer->GetParamAsInts("pads_end"));
}

}  // namespace vpu
