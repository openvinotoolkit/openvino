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
#include <ngraph/opsets/opset4.hpp>

using namespace InferenceEngine;

namespace vpu {
VPU_DECLARE_ENUM(InterpolateMode,
    nearest = 0,
    linear = 1,
    linear_onnx = 2,
    cubic = 3
)
VPU_DECLARE_ENUM(InterpolateShapeCalcMode,
    sizes = 0,
    scales = 1
)
VPU_DECLARE_ENUM(InterpolateCoordTransMode,
    half_pixel = 0,
    pytorch_half_pixel = 1,
    asymmetric = 2,
    tf_half_pixel_for_nn = 3,
    align_corners = 4
)
VPU_DECLARE_ENUM(InterpolateNearestMode,
    round_prefer_floor = 0,
    round_prefer_ceil = 1,
    floor = 2,
    ceil = 3,
    simple = 4
)

namespace {
class InterpolateStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<InterpolateStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        printf("propagateDataOrderImpl start\n");
        auto input0 = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

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
        printf("initialCheckImpl start\n");
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::S32}, {DataType::FP16}, {DataType::S32}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto sampleType = attrs().get<InterpolateMode>("mode");
        auto sampleShapeCalcMode = attrs().get<InterpolateShapeCalcMode>("shape_calculation_mode");
        auto& pads_begin = attrs().get<std::vector<size_t>>("pads_begin");
        auto& pads_end = attrs().get<std::vector<size_t>>("pads_end");
        auto sampleCoordTransMode = attrs().get<InterpolateCoordTransMode>("coordinate_transformation_mode");
        auto sampleNearestMode = attrs().get<InterpolateNearestMode>("nearest_mode");
        auto cube_coeff = attrs().get<float>("cube_coeff");
        auto antialias = attrs().get<bool>("antialias");

        serializer.append(static_cast<InterpolateMode>(sampleType));
        serializer.append(static_cast<InterpolateShapeCalcMode>(sampleShapeCalcMode));
        serializer.append(static_cast<std::vector<size_t>>(pads_begin));
        serializer.append(static_cast<std::vector<size_t>>(pads_end));
        serializer.append(static_cast<InterpolateCoordTransMode>(sampleCoordTransMode));
        serializer.append(static_cast<InterpolateNearestMode>(sampleNearestMode));
        serializer.append(static_cast<float>(cube_coeff));
        serializer.append(static_cast<int>(antialias));
        printf("serializeParamsImpl end\n");
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        printf("serializeDataImpl start\n");
        for (int i = 0; i < numInputs(); i++) {
            inputEdge(i)->input()->serializeBuffer(serializer);
        }
        outputEdge(0)->output()->serializeBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addInterpolateStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const DataVector& input,
        const DataVector& output) {
    Stage interpolateStage = model->addNewStage<InterpolateStage>(
        name,
        StageType::Interpolate,
        layer,
        {input[0], input[1], input[2], input[3]},
        {output});

    return interpolateStage;
}

void FrontEnd::parseInterpolate(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    printf("PARSE start\n");
    IE_ASSERT(inputs.size() == 4);
    IE_ASSERT(outputs.size() == 1);

    ie::details::CaselessEq<std::string> cmp;

    auto stage = model->addNewStage<InterpolateStage>(layer->name, StageType::Interpolate, layer, {inputs[0], inputs[1], inputs[2], inputs[3]}, {outputs});
    stage->attrs().set<int>("antialias", layer->GetParamAsBool("antialias", 0));

    auto mode = layer->GetParamAsString("mode", "nearest");
    auto nearest_m = layer->GetParamAsString("nearest_mode", "round_prefer_floor");
    auto shape_calc = layer->GetParamAsString("shape_calculation_mode", "scales");
    auto coord_trans = layer->GetParamAsString("coordinate_transformation_mode", "half_pixel");

    if (cmp(mode, "nearest")) {
        stage->attrs().set<InterpolateMode>("mode", InterpolateMode::nearest);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " supports only nearest mode";
    }
    if (cmp(nearest_m, "round_prefer_floor")) {
        stage->attrs().set<InterpolateNearestMode>("nearest_mode", InterpolateNearestMode::round_prefer_floor);
    } else if (cmp(nearest_m, "round_prefer_ceil")) {
        stage->attrs().set<InterpolateNearestMode>("nearest_mode", InterpolateNearestMode::round_prefer_ceil);
    } else if (cmp(nearest_m, "floor")) {
        stage->attrs().set<InterpolateNearestMode>("nearest_mode", InterpolateNearestMode::floor);
    } else if (cmp(nearest_m, "ceil")) {
        stage->attrs().set<InterpolateNearestMode>("nearest_mode", InterpolateNearestMode::ceil);
    } else if (cmp(nearest_m, "simple")) {
        stage->attrs().set<InterpolateNearestMode>("nearest_mode", InterpolateNearestMode::simple);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " does not support this nearest mode";
    }

    if (cmp(shape_calc, "scales")) {
        stage->attrs().set<InterpolateShapeCalcMode>("shape_calculation_mode", InterpolateShapeCalcMode::scales);
    } else if (cmp(shape_calc, "sizes")) {
        stage->attrs().set<InterpolateShapeCalcMode>("shape_calculation_mode", InterpolateShapeCalcMode::sizes);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " does not support this shape calculation mode";
    }

    if (cmp(coord_trans, "half_pixel")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordinate_transformation_mode", InterpolateCoordTransMode::half_pixel);
    } else if (cmp(coord_trans, "pytorch_half_pixel")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordinate_transformation_mode", InterpolateCoordTransMode::pytorch_half_pixel);
    } else if (cmp(coord_trans, "asymmetric")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordinate_transformation_mode", InterpolateCoordTransMode::asymmetric);
    } else if (cmp(coord_trans, "tf_half_pixel_for_nn")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordinate_transformation_mode", InterpolateCoordTransMode::tf_half_pixel_for_nn);
    } else if (cmp(coord_trans, "align_corners")) {
        stage->attrs().set<InterpolateCoordTransMode>("coordinate_transformation_mode", InterpolateCoordTransMode::align_corners);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " does not support this coordinate transformation mode";
    }
    stage->attrs().set<float>("cube_coeff", layer->GetParamAsFloat("cube_coeff", 0));
    // stage->attrs().set<std::vector<int>>("pads_begin", layer->GetParamAsInts("pads_begin"));
    // stage->attrs().set<std::vector<int>>("pads_end", layer->GetParamAsInts("pads_end"));
    printf("PARSE end\n");
}

}  // namespace vpu
