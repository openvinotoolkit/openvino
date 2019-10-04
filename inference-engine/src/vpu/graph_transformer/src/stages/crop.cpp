// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>

namespace vpu {

namespace {

class CropStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<CropStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step,
            StageDataInfo<float>& scaleInfo) override {
        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales[0];
            scaleInfo.setOutput(outputEdge(0), inputScale);
        } else {
            // Crop can only propagate scaling, not generate.

            for (const auto& inEdge : inputEdges()) {
                scaleInfo.setInput(inEdge, 1.0f);
            }
            scaleInfo.setOutput(outputEdge(0), 1.0f);
        }
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        auto inOrder = input->desc().dimsOrder();

        // HWC only
        orderInfo.setInput(inputEdge(0), inOrder.createMovedDim(Dim::C, 0));
        orderInfo.setOutput(outputEdge(0), inOrder.createMovedDim(Dim::C, 0));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        for (const auto& inEdge : inputEdges()) {
            batchInfo.setInput(inEdge, BatchSupport::Split);
        }
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 1 || numInputs() == 2);
        IE_ASSERT(numOutputs() == 1);
        assertAllInputsOutputsTypes(this, DataType::FP16, DataType::FP16);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& offset = attrs().get<DimValues>("offset");

        serializer.append(static_cast<int32_t>(offset.get(Dim::W, 0)));
        serializer.append(static_cast<int32_t>(offset.get(Dim::H, 0)));
        serializer.append(static_cast<int32_t>(offset.get(Dim::C, 0)));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeOldBuffer(handle_from_this(), serializer);
        output->serializeOldBuffer(handle_from_this(), serializer);
    }
};

}  // namespace

void FrontEnd::parseCrop(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    // TODO : Crop layer in IR might have 1 or 2 inputs
    IE_ASSERT(inputs.size() >= 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::CropLayer>(_layer);
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(layer->axis.size() == layer->offset.size());

    auto cropAxis = layer->axis[0];
    if (cropAxis < 0) {
        cropAxis += 4;
    }

    if (cropAxis < 0 || cropAxis > 3) {
        VPU_THROW_EXCEPTION
            << "Layer " << layer->name << " [" << layer->type
            << "] has invalid axis value. Expected: 0 <= axis < 4, Actual: " << cropAxis;
    }

    auto stage = model->addNewStage<CropStage>(
        layer->name,
        StageType::Crop,
        layer,
        inputs,
        outputs);

    DimValues offset;
    for (int i = 0; i < layer->offset.size(); i++) {
        offset.set(static_cast<Dim>(3 - cropAxis - i), layer->offset[i]);
    }

    stage->attrs().set("offset", offset);
}

}  // namespace vpu
