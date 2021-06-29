// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/interpolate_stages.hpp>

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
        auto align_corners = attrs().get<bool>(g_antialias);
        auto factor = attrs().get<float>(g_factor);
        auto sampleType = attrs().get<ResampleType>(g_type);
        auto coordinateTransformationMode = attrs().get<InterpolateCoordTransMode>(g_coordinate_transformation_mode);
        auto nearestMode = attrs().get<InterpolateNearestMode>(g_nearest_mode);

        serializer.append(static_cast<int32_t>(align_corners));
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
            const NodePtr& node,
            bool antialias,
            InterpolateCoordTransMode coordinateTransformationMode,
            InterpolateNearestMode nearestMode,
            float factor,
            const Data& input,
            const Data& output) {
    auto stage = model->addNewStage<ResampleStage>(node->get_name(), StageType::Resample, node, {input}, {output});

    stage->attrs().set<bool>(g_antialias, antialias);
    stage->attrs().set<InterpolateCoordTransMode>(g_coordinate_transformation_mode, coordinateTransformationMode);
    stage->attrs().set<InterpolateNearestMode>(g_nearest_mode, nearestMode);
    stage->attrs().set<float>(g_factor, factor);
    stage->attrs().set<ResampleType>(g_type, ResampleType::Nearest);

    return stage;
}
// resample/interpolate case
void FrontEnd::parseResample(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    // VPU_THROW_UNLESS(inputs.size() == 1,
    //                  "Resample stage with name {} must have only 1 input, "
    //                  "actually provided {}", node->get_friendly_name(), inputs.size());
    // VPU_THROW_UNLESS(outputs.size() == 1,
    //                  "Resample stage with name {} must have only 1 output, "
    //                  "actually provided {}", layer->name, outputs.size());

    // ie::details::CaselessEq<std::string> cmp;
    // const auto method  = layer->GetParamAsString(g_type, "caffe.ResampleParameter.NEAREST");
    // const auto coord   = layer->GetParamAsString(g_coordinate_transformation_mode, g_half_pixel);
    // const auto nearest = layer->GetParamAsString(g_nearest_mode, g_round_prefer_ceil);

    // const auto coordModeIt   = coordTransformModeMap.find(coord);
    // const auto nearestModeIt = nearestModeMap.find(nearest);
    // VPU_THROW_UNLESS(coordModeIt != coordTransformModeMap.end(), "Resample stage does not support this coordinate transforation mode");
    // VPU_THROW_UNLESS(nearestModeIt != nearestModeMap.end(), "Resample stage does not support this nearest transforation mode");
    // auto coordinateTransformationMode = coordModeIt->second;
    // auto nearestMode = nearestModeIt->second;

    // if (cmp(method, "caffe.ResampleParameter.NEAREST")) {
    //     _stageBuilder->addResampleNearestStage(model,
    //                                            layer->name,
    //                                            layer,
    //                                            layer->GetParamAsInt(g_antialias, 0),
    //                                            coordinateTransformationMode, nearestMode,
    //                                            layer->GetParamAsFloat(g_factor, -1),
    //                                            inputs[0],
    //                                            outputs[0]);
    // } else {
    //     VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " supports only caffe.ResampleParameter.NEAREST resample type";
    // }
}

}  // namespace vpu
