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
    auto stage = model->addNewStage<ResampleStage>(node->get_friendly_name(), StageType::Resample, node, {input}, {output});

    stage->attrs().set<bool>(g_antialias, antialias);
    stage->attrs().set<InterpolateCoordTransMode>(g_coordinate_transformation_mode, coordinateTransformationMode);
    stage->attrs().set<InterpolateNearestMode>(g_nearest_mode, nearestMode);
    stage->attrs().set<float>(g_factor, factor);
    stage->attrs().set<ResampleType>(g_type, ResampleType::Nearest);

    return stage;
}

}  // namespace vpu
