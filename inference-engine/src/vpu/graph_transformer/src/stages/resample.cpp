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
        auto antialias = attrs().get<bool>("antialias");
        auto factor = attrs().get<float>("factor");
        auto sampleType = attrs().get<ResampleType>("type");
        auto coordinateTransformationMode = attrs().get<InterpolateCoordTransMode>("coordinate_transformation_mode");
        auto nearestMode = attrs().get<InterpolateNearestMode>("nearest_mode");

        serializer.append(static_cast<int32_t>(antialias));
        serializer.append(static_cast<float>(factor));
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
            bool antialias,
            InterpolateCoordTransMode coordinateTransformationMode,
            InterpolateNearestMode nearestMode,
            float factor,
            const Data& input,
            const Data& output) {
    auto stage = model->addNewStage<ResampleStage>(layer->name, StageType::Resample, layer, {input}, {output});

    stage->attrs().set<bool>("antialias", antialias);
    stage->attrs().set<InterpolateCoordTransMode>("coordinate_transformation_mode", coordinateTransformationMode);
    stage->attrs().set<InterpolateNearestMode>("nearest_mode", nearestMode);
    stage->attrs().set<float>("factor", factor);
    stage->attrs().set<ResampleType>("type", ResampleType::Nearest);

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
    const auto method = layer->GetParamAsString("type", "caffe.ResampleParameter.NEAREST");
    const auto coord = layer->GetParamAsString("coordinate_transformation_mode", "half_pixel");
    const auto nearest = layer->GetParamAsString("nearest_mode", "round_prefer_ceil");
    InterpolateCoordTransMode coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
    InterpolateNearestMode nearestMode = InterpolateNearestMode::RoundPreferCeil;

    if (cmp(coord, "asymmetric")) {
        coordinateTransformationMode = InterpolateCoordTransMode::Asymmetric;
    }
    if (cmp(nearest, "floor")) {
        nearestMode = InterpolateNearestMode::Floor;
    }

    if (cmp(method, "caffe.ResampleParameter.NEAREST")) {
        _stageBuilder->addResampleNearestStage(model,
                                               layer->name,
                                               layer,
                                               layer->GetParamAsInt("antialias", 0),
                                               coordinateTransformationMode, nearestMode,
                                               layer->GetParamAsFloat("factor", -1),
                                               inputs[0],
                                               outputs[0]);
    } else {
        VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " supports only caffe.ResampleParameter.NEAREST resample type";
    }
}

}  // namespace vpu
