// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/interpolate_stages.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class InterpStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<InterpStage>(*this);
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
        auto align_corners = attrs().get<bool>(g_align_corners);
        auto sampleType = attrs().get<InterpolateMode>(g_mode);
        auto coordinateTransMode = attrs().get<InterpolateCoordTransMode>(g_coordinate_transformation_mode);

        serializer.append(static_cast<int32_t>(align_corners));
        serializer.append(static_cast<uint32_t>(sampleType));
        serializer.append(static_cast<uint32_t>(coordinateTransMode));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addInterpStage(
                        const Model& model,
                        const std::string& name,
                        const ie::CNNLayerPtr& layer,
                        bool align_corners,
                        InterpolateMode mode,
                        InterpolateCoordTransMode coordinateTransMode,
                        const Data& input,
                        const Data& output) {
    auto stage = model->addNewStage<InterpStage>(layer->name, StageType::Interp, layer, {input}, {output});
    stage->attrs().set<bool>(g_align_corners, align_corners);
    stage->attrs().set<InterpolateMode>(g_mode, mode);
    stage->attrs().set<InterpolateCoordTransMode>(g_coordinate_transformation_mode, coordinateTransMode);

    return stage;
}

void FrontEnd::parseInterp(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Interp stage with name {} must have only 1 input, "
                     "actually provided {}", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Interp stage with name {} must have only 1 output, "
                     "actually provided {}", layer->name, outputs.size());
    const auto coord = layer->GetParamAsString(g_coordinate_transformation_mode, g_half_pixel);
    const auto interpMode = layer->GetParamAsString(g_mode, g_linear);
    const auto interpModeIt = interpModeMap.find(interpMode);
    const auto coordModeIt  = coordTransformModeMap.find(coord);
    VPU_THROW_UNLESS(interpModeIt != interpModeMap.end(), "Interp stage with name {} does not support this interp mode", layer->name);
    VPU_THROW_UNLESS(interpModeIt->second == InterpolateMode::Linear || interpModeIt->second  == InterpolateMode::LinearOnnx,
                     "Interp stage supports linear and linear_onnx modes");
    VPU_THROW_UNLESS(coordModeIt != coordTransformModeMap.end(), "Interp stage does not support this coordinate transforation mode");
    auto coordinateTransMode = coordModeIt->second;
    auto mode = interpModeIt->second;

    _stageBuilder->addInterpStage(model, layer->name, layer, layer->GetParamAsInt(g_align_corners, 0), mode, coordinateTransMode, inputs[0], outputs[0]);
}

}  // namespace vpu
