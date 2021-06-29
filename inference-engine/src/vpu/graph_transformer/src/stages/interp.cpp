// Copyright (C) 2018-2021 Intel Corporation
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
                        const NodePtr& node,
                        bool align_corners,
                        InterpolateMode mode,
                        InterpolateCoordTransMode coordinateTransMode,
                        const Data& input,
                        const Data& output) {
    auto stage = model->addNewStage<InterpStage>(node->get_name(), StageType::Interp, node, {input}, {output});
    stage->attrs().set<bool>(g_align_corners, align_corners);
    stage->attrs().set<InterpolateMode>(g_mode, mode);
    stage->attrs().set<InterpolateCoordTransMode>(g_coordinate_transformation_mode, coordinateTransMode);

    return stage;
}

// need to rework logic
void FrontEnd::parseInterp(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    // const auto& interp = ngraph::as_type_ptr<ngraph::op::v4::Interpolate>(node);
    // IE_ASSERT(interp != nullptr);
    // VPU_THROW_UNLESS(inputs.size() == 1,
    //                  "Interp stage with name {} must have only 1 input, "
    //                  "actually provided {}", interp->get_name(), inputs.size());
    // VPU_THROW_UNLESS(outputs.size() == 1,
    //                  "Interp stage with name {} must have only 1 output, "
    //                  "actually provided {}", interp->get_name(), outputs.size());
    // const auto attrs = interp->get_attrs();
    // const auto coord = attrs.coordinate_transformation_mode;
    // const auto interpMode = attrs.mode;
    // // const auto interpModeIt = interpModeMap.find(interpMode);  ??
    // // const auto coordModeIt  = attrs.coordinate_transformation_mode; ??
    // VPU_THROW_UNLESS(interpMode == ngraph::op::v4::Interpolate::InterpolateMode::linear ||
    //                  interpMode == ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
    //                  "Interp stage supports linear and linear_onnx modes");
    // auto coordinateTransMode = coordModeIt->second;
    // auto mode = interpModeIt->second;

    // _stageBuilder->addInterpStage(model, node->get_friendly_name(), node, layer->GetParamAsInt(g_align_corners, 0), mode, coordinateTransMode, inputs[0], outputs[0]);
}

}  // namespace vpu
