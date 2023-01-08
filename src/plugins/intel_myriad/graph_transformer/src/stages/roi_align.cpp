// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

VPU_DECLARE_ENUM(ROIAlignMode,
    Average = 0,
    Max = 1
)

VPU_DECLARE_ENUM(ROIAlignStep,
    Repacking = 0,
    ROIAlignCHWc = 1,
    ROIAlign = 2
)

static const char s_mode[] = "mode";
static const char s_pooled_w[] = "pooled_w";
static const char s_pooled_h[] = "pooled_h";
static const char s_sampling_ratio[] = "sampling_ratio";
static const char s_spatial_scale[] = "spatial_scale";
static const char s_step_number[] = "step_number";

namespace {

class ROIAlignStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ROIAlignStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        orderInfo.setInput(inputEdge(0), inputEdge(0)->input()->desc().dimsOrder().createMovedDim(Dim::C, 2));
        orderInfo.setOutput(outputEdge(0), outputEdge(0)->output()->desc().dimsOrder().createMovedDim(Dim::C, 2));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        for (const auto& inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        const auto step_number = attrs().get<ROIAlignStep>(s_step_number);
        std::vector<EnumSet<DataType>> repackingInputs = {{DataType::FP16}};
        std::vector<EnumSet<DataType>> ROIAlignInputs = {{DataType::FP16}, {DataType::FP16}, {DataType::S32}};

        assertInputsOutputsTypes(this, (step_number == ROIAlignStep::Repacking) ? repackingInputs : ROIAlignInputs, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto pooled_w = attrs().get<int>(s_pooled_w);
        const auto pooled_h = attrs().get<int>(s_pooled_h);
        const auto sampling_ratio = attrs().get<int>(s_sampling_ratio);
        const auto spatial_scale = attrs().get<float>(s_spatial_scale);
        const auto mode = attrs().get<ROIAlignMode>(s_mode);
        const auto step_number = attrs().get<ROIAlignStep>(s_step_number);

        serializer.append(static_cast<uint32_t>(pooled_w));
        serializer.append(static_cast<uint32_t>(pooled_h));
        serializer.append(static_cast<uint32_t>(sampling_ratio));
        serializer.append(static_cast<float>(spatial_scale));
        serializer.append(static_cast<ROIAlignMode>(mode));
        serializer.append(static_cast<ROIAlignStep>(step_number));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        for (int i = 0; i < numInputs(); i++) {
            inputEdge(i)->input()->serializeBuffer(serializer);
        }

        outputEdge(0)->output()->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseROIAlign(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 3 || inputs.size() == 1,
                    "ROIAlign stage with name {} has invalid number of inputs: expected 3 or 1 "
                    "actually provided {}", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                    "ROIAlign stage with name {} has invalid number of outputs: expected 1, "
                    "actually provided {}", layer->name, outputs.size());

    const auto mode = layer->GetParamAsString("mode", "");
    VPU_THROW_UNLESS(mode == "avg" || mode == "max", "Layer with name {} supports only (avg, max) mode", layer->name);
    ROIAlignMode roi_align_mode = (mode == "avg") ? ROIAlignMode::Average : ROIAlignMode::Max;

    const auto width = inputs[0]->desc().dim(Dim::W);
    const auto is_input_static = (inputs[0]->parentDataToShapeEdge() == nullptr);
    const auto use_chwc_repacking = (width >= 200) && (roi_align_mode == ROIAlignMode::Average) && is_input_static;
    auto repackedInput = inputs[0];

    if (use_chwc_repacking) {
        repackedInput = model->duplicateData(inputs[0], formatString("@ROIAlignRepacked"));

        const auto repacking_stage = model->addNewStage<ROIAlignStage>(layer->name + "Repacking",
                                                                       StageType::ROIAlign, layer,
                                                                       {inputs[0]}, {repackedInput});

        repacking_stage->attrs().set<int>(s_pooled_w, layer->GetParamAsInt("pooled_w"));
        repacking_stage->attrs().set<int>(s_pooled_h, layer->GetParamAsInt("pooled_h"));
        repacking_stage->attrs().set<int>(s_sampling_ratio, layer->GetParamAsInt("sampling_ratio"));
        repacking_stage->attrs().set<float>(s_spatial_scale, layer->GetParamAsFloat("spatial_scale"));
        repacking_stage->attrs().set<ROIAlignMode>(s_mode, ROIAlignMode::Average);
        repacking_stage->attrs().set<ROIAlignStep>(s_step_number, ROIAlignStep::Repacking);
    }

    const auto stage = model->addNewStage<ROIAlignStage>(layer->name, StageType::ROIAlign, layer, {repackedInput, inputs[1], inputs[2]}, outputs);

    stage->attrs().set<ROIAlignMode>(s_mode, roi_align_mode);
    stage->attrs().set<int>(s_pooled_w, layer->GetParamAsInt("pooled_w"));
    stage->attrs().set<int>(s_pooled_h, layer->GetParamAsInt("pooled_h"));
    stage->attrs().set<int>(s_sampling_ratio, layer->GetParamAsInt("sampling_ratio"));
    stage->attrs().set<float>(s_spatial_scale, layer->GetParamAsFloat("spatial_scale"));
    stage->attrs().set<ROIAlignStep>(s_step_number, use_chwc_repacking ? ROIAlignStep::ROIAlignCHWc : ROIAlignStep::ROIAlign);
}

}  // namespace vpu
