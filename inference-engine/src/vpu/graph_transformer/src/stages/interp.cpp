// Copyright (C) 2018-2020 Intel Corporation
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
        auto align = attrs().get<bool>(g_align_corners);
        auto sampleType = attrs().get<InterpolateMode>(g_mode);
        auto coordinateTransMode = attrs().get<InterpolateCoordTransMode>(g_coordinate_transformation_mode);

        serializer.append(static_cast<int32_t>(align));
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
                        bool align,
                        InterpolateMode mode,
                        InterpolateCoordTransMode coordinateTransMode,
                        const Data& input,
                        const Data& output) {
    auto stage = model->addNewStage<InterpStage>(layer->name, StageType::Interp, layer, {input}, {output});
    stage->attrs().set<bool>(g_align_corners, align);
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
    ie::details::CaselessEq<std::string> cmp;
    const auto coord = layer->GetParamAsString(g_coordinate_transformation_mode, g_half_pixel);
    const auto interpMode = layer->GetParamAsString(g_mode, g_linear);
    InterpolateCoordTransMode coordinateTransMode = InterpolateCoordTransMode::HalfPixel;
    InterpolateMode mode = InterpolateMode::Linear;

    if (cmp(coord, g_asymmetric)) {
        coordinateTransMode = InterpolateCoordTransMode::Asymmetric;
    } else if (cmp(coord, g_half_pixel)) {
        coordinateTransMode = InterpolateCoordTransMode::HalfPixel;
    } else if (cmp(coord, g_pytorch_half_pixel)) {
        coordinateTransMode = InterpolateCoordTransMode::PytorchHalfPixel;
    } else if (cmp(coord, g_tf_half_pixel_for_nn)) {
        coordinateTransMode = InterpolateCoordTransMode::TfHalfPixelForNn;
    } else if (cmp(coord, g_align_corners)) {
        coordinateTransMode = InterpolateCoordTransMode::AlignCorners;
    } else {
        VPU_THROW_FORMAT("Current Interp doesn't support this coordinate transformation mode");
    }

    if (cmp(interpMode, g_linear_onnx)) {
        mode = InterpolateMode::LinearOnnx;
    }

    _stageBuilder->addInterpStage(model, layer->name, layer, layer->GetParamAsInt(g_align_corners, 0), mode, coordinateTransMode, inputs[0], outputs[0]);
}

}  // namespace vpu
