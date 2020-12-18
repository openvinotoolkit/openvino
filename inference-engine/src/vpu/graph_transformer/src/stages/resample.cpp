// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

constexpr auto coordinate_transformation_mode = "coordinate_transformation_mode";
constexpr auto mode                 = "mode";
constexpr auto asymmetric           = "asymmetric";
constexpr auto half_pixel           = "half_pixel";
constexpr auto nearest_mode         = "nearest_mode";
constexpr auto round_prefer_floor   = "round_prefer_floor";
constexpr auto round_prefer_ceil    = "round_prefer_ceil";
constexpr auto floor_mode           = "floor";
constexpr auto antialias            = "antialias";
constexpr auto factor               = "factor";
constexpr auto typeI                = "type";

namespace vpu {

VPU_DECLARE_ENUM(ResampleType,
    Nearest  = 0,  // Currently this is only one supported
    Linear = 1,
    Cubic = 2
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
        auto antial = attrs().get<bool>(antialias);
        auto fact = attrs().get<float>(factor);
        auto sampleType = attrs().get<ResampleType>(typeI);
        auto coordinateTransformationMode = attrs().get<InterpolateCoordTransMode>(coordinate_transformation_mode);
        auto nearestMode = attrs().get<InterpolateNearestMode>(nearest_mode);

        serializer.append(static_cast<int32_t>(antial));
        serializer.append(static_cast<float>(fact));
        serializer.append(static_cast<uint32_t>(sampleType));
        serializer.append(static_cast<uint32_t>(coordinateTransformationMode));
        serializer.append(static_cast<uint32_t>(nearestMode));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addResampleNearestStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            bool antial,
            InterpolateCoordTransMode coordinateTransformationMode,
            InterpolateNearestMode nearestMode,
            float fact,
            const Data& input,
            const Data& output) {
    auto stage = model->addNewStage<ResampleStage>(layer->name, StageType::Resample, layer, {input}, {output});

    stage->attrs().set<bool>(antialias, antial);
    stage->attrs().set<InterpolateCoordTransMode>(coordinate_transformation_mode, coordinateTransformationMode);
    stage->attrs().set<InterpolateNearestMode>(nearest_mode, nearestMode);
    stage->attrs().set<float>(factor, fact);
    stage->attrs().set<ResampleType>(typeI, ResampleType::Nearest);

    return stage;
}

void FrontEnd::parseResample(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Resample stage with name {} must have only 1 input, "
                     "actually provided {}", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Resample stage with name {} must have only 1 output, "
                     "actually provided {}", layer->name, outputs.size());

    ie::details::CaselessEq<std::string> cmp;
    const auto method  = layer->GetParamAsString(typeI, "caffe.ResampleParameter.NEAREST");
    const auto coord   = layer->GetParamAsString(coordinate_transformation_mode, half_pixel);
    const auto nearest = layer->GetParamAsString(nearest_mode, round_prefer_ceil);
    InterpolateCoordTransMode coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
    InterpolateNearestMode nearestMode = InterpolateNearestMode::RoundPreferCeil;

    if (cmp(coord, asymmetric)) {
        coordinateTransformationMode = InterpolateCoordTransMode::Asymmetric;
    }
    if (cmp(nearest, floor_mode)) {
        nearestMode = InterpolateNearestMode::Floor;
    }

    if (cmp(method, "caffe.ResampleParameter.NEAREST")) {
        _stageBuilder->addResampleNearestStage(model,
                                               layer->name,
                                               layer,
                                               layer->GetParamAsInt(antialias, 0),
                                               coordinateTransformationMode, nearestMode,
                                               layer->GetParamAsFloat(factor, -1),
                                               inputs[0],
                                               outputs[0]);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " supports only caffe.ResampleParameter.NEAREST resample type";
    }
}

}  // namespace vpu
