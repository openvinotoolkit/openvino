// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

namespace vpu {

VPU_DECLARE_ENUM(ResampleType,
    Nearest  = 0,  // Currently this is only one supported
    Linear = 1,
    Cubic = 2
)

VPU_DECLARE_ENUM(ResampleCoordTransMode,
    // the coordinate in the original tensor axis `x` is calculated as `((x_resized + 0.5) / scale[x]) - 0.5`
    half_pixel = 0,
    // the coordinate in the original tensor axis `x` is calculated by `(x_resized + 0.5) / scale[x] - 0.5 if output_shape[x] > 1 else 0.0`
    pytorch_half_pixel = 1,
    // the coordinate in the original tensor axis `x` is calculated according to the formula `x_resized / scale[x]`
    asymmetric = 2,
    // the coordinate in the original tensor axis `x` is `(x_resized + 0.5) / scale[x]`
    tf_half_pixel_for_nn = 3,
    // the coordinate in the original tensor axis `x` is calculated as `0 if output_shape[x] == 1 else x_resized * (input_shape[x] - 1) / (output_shape[x] - 1)`
    align_corners = 4
)

VPU_DECLARE_ENUM(ResampleNearestMode,
    // this mode is known as round half down
    round_prefer_floor = 0,
    // it is round half up mode
    round_prefer_ceil = 1,
    // this mode computes the largest integer value not greater than rounded value
    floor = 2,
    // this mode computes the smallest integer value not less than rounded value
    ceil = 3,
    // this mode behaves as `ceil` mode when `Interpolate` is downsample, and as dropping the fractional part otherwise
    simple = 4
)

VPU_DECLARE_ENUM(ResampleShapeCalcMode,
    // nearest neighbor 
    sizes = 0,
    // cubic interpolation
    scales = 1
)

VPU_DECLARE_ENUM(ResampleAxis,
    along_b = 0,
    along_f = 1,
    along_x = 2,
    along_y = 3,
    along_z = 4,
    along_w = 5
)

namespace {

class ResampleStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ResampleStage>(*this);
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
        auto axis = attrs().get<ResampleAxis>("resampleAxis");
        auto sampleType = attrs().get<ResampleType>("type");
        auto sampleNearestMode = attrs().get<ResampleNearestMode>("nearestMode");
        auto sampleShapeCalcMode = attrs().get<ResampleShapeCalcMode>("shapeCalcMode");
        auto sampleCoordTransMode = attrs().get<ResampleCoordTransMode>("coordTransMode");

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
    int32_t pad_begin;
    int32_t antialias;
    int32_t align_corners;

    float cube_coeff;

    ResampleType operation_type;
    ResampleNearestMode round_mode;
    ResampleShapeCalcMode shape_calc_mode;
    ResampleCoordTransMode coord_trans_mode;
};

}  // namespace

void FrontEnd::parseResample(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    ie::details::CaselessEq<std::string> cmp;

    auto stage = model->addNewStage<ResampleStage>(layer->name, StageType::Resample, layer, inputs, outputs);

    stage->attrs().set<bool>("antialias", layer->GetParamAsInt("antialias", 0));
    stage->attrs().set<float>("factor", layer->GetParamAsFloat("factor", -1.0f));

    auto method = layer->GetParamAsString("type", "caffe.ResampleParameter.NEAREST");
    if (cmp(method, "caffe.ResampleParameter.NEAREST")) {
        stage->attrs().set<ResampleType>("type", ResampleType::Nearest);
    } else if (cmp(method, "caffe.ResampleParameter.LINEAR")) {
        stage->attrs().set<ResampleType>("type", ResampleType::Linear);
    } else if (cmp(method, "caffe.ResampleParameter.CUBIC")) {
        stage->attrs().set<ResampleType>("type", ResampleType::Cubic);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this resample type";
    }

    auto nnMode = layer->GetParamAsString("nearestMode", "caffe.ResampleParameter.round_prefer_floor");
    if (cmp(nnMode, "caffe.ResampleParameter.round_prefer_floor")) {
        stage->attrs().set<ResampleNearestMode>("nearestMode", ResampleNearestMode::round_prefer_floor);
    } else if (cmp(nnMode, "caffe.ResampleParameter.round_prefer_ceil")) {
        stage->attrs().set<ResampleNearestMode>("nearestMode", ResampleNearestMode::round_prefer_ceil);
    } else if (cmp(nnMode, "caffe.ResampleParameter.floor")) {
        stage->attrs().set<ResampleNearestMode>("nearestMode", ResampleNearestMode::floor);
    } else if (cmp(nnMode, "caffe.ResampleParameter.ceil")) {
        stage->attrs().set<ResampleNearestMode>("nearestMode", ResampleNearestMode::ceil);
    } else if (cmp(nnMode, "caffe.ResampleParameter.simple")) {
        stage->attrs().set<ResampleNearestMode>("nearestMode", ResampleNearestMode::simple);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this resample nearest mode variant";
    }

    auto resampleAxis = layer->GetParamAsString("resampleAxis", "caffe.ResampleParameter.along_b");
    if (cmp(nnMode, "caffe.ResampleParameter.along_b")) {
        stage->attrs().set<ResampleNearestMode>("resampleAxis", ResampleAxis::along_b);
    } else if (cmp(nnMode, "caffe.ResampleParameter.along_f")) {
        stage->attrs().set<ResampleNearestMode>("resampleAxis", ResampleAxis::along_f);
    } else if (cmp(nnMode, "caffe.ResampleParameter.along_x")) {
        stage->attrs().set<ResampleNearestMode>("resampleAxis", ResampleAxis::along_x);
    } else if (cmp(nnMode, "caffe.ResampleParameter.along_y")) {
        stage->attrs().set<ResampleNearestMode>("resampleAxis", ResampleAxis::along_y);
    } else if (cmp(nnMode, "caffe.ResampleParameter.along_z")) {
        stage->attrs().set<ResampleNearestMode>("resampleAxis", ResampleAxis::along_z);
    } else if (cmp(nnMode, "caffe.ResampleParameter.along_w")) {
        stage->attrs().set<ResampleNearestMode>("resampleAxis", ResampleAxis::along_w);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this axis";
    }

    auto shapeCalcMode = layer->GetParamAsString("shapeCalcMode", "caffe.ResampleParameter.sizes");
    if (cmp(shapeCalcMode, "caffe.ResampleParameter.sizes")) {
        stage->attrs().set<ResampleShapeCalcMode>("shapeCalcMode", ResampleShapeCalcMode::sizes);
    } else if (cmp(shapeCalcMode, "caffe.ResampleParameter.scales")) {
        stage->attrs().set<ResampleShapeCalcMode>("shapeCalcMode", ResampleShapeCalcMode::scales);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this resample shape calculation mode";
    }

    auto coordTransMode = layer->GetParamAsString("coordTransMode", "caffe.ResampleParameter.half_pixel");
    if (cmp(coordTransMode, "caffe.ResampleParameter.half_pixel")) {
        stage->attrs().set<ResampleCoordTransMode>("coordTransMode", ResampleCoordTransMode::half_pixel);
    } else if (cmp(coordTransMode, "caffe.ResampleParameter.pytorch_half_pixel")) {
        stage->attrs().set<ResampleCoordTransMode>("coordTransMode", ResampleCoordTransMode::pytorch_half_pixel);
    } else if (cmp(coordTransMode, "caffe.ResampleParameter.pytorch_half_pixel")) {
        stage->attrs().set<ResampleCoordTransMode>("coordTransMode", ResampleCoordTransMode::pytorch_half_pixel);
    } else if (cmp(coordTransMode, "caffe.ResampleParameter.tf_half_pixel_for_nn")) {
        stage->attrs().set<ResampleCoordTransMode>("coordTransMode", ResampleCoordTransMode::tf_half_pixel_for_nn);
    } else if (cmp(coordTransMode, "caffe.ResampleParameter.align_corners")) {
        stage->attrs().set<ResampleCoordTransMode>("coordTransMode", ResampleCoordTransMode::align_corners);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this resample coordinate transform mode";
    }
}

}  // namespace vpu
