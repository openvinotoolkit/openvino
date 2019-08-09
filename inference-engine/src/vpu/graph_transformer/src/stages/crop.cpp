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
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() >= 1);
        IE_ASSERT(_outputEdges.size() == 1);

        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales[0];
            _scaleInfo.setOutput(_outputEdges[0], inputScale);
        } else {
            // Crop can only propagate scaling, not generate.

            for (const auto& inEdge : _inputEdges) {
                _scaleInfo.setInput(inEdge, 1.0f);
            }
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() >= 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        auto inOrder = input->desc().dimsOrder();

        // HWC only
        _orderInfo.setInput(_inputEdges[0], inOrder.createMovedDim(Dim::C, 0));
        _orderInfo.setOutput(_outputEdges[0], inOrder.createMovedDim(Dim::C, 0));
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() >= 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _stridesInfo.setInput(_inputEdges[0], StridesRequirement::compact());
        _stridesInfo.setOutput(_outputEdges[0], StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() >= 1);
        IE_ASSERT(_outputEdges.size() == 1);

        for (const auto& inEdge : _inputEdges) {
            _batchInfo.setInput(inEdge, BatchSupport::Split);
        }
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& offset = attrs().get<DimValues>("offset");

        serializer.append(static_cast<int32_t>(offset.get(Dim::W, 0)));
        serializer.append(static_cast<int32_t>(offset.get(Dim::H, 0)));
        serializer.append(static_cast<int32_t>(offset.get(Dim::C, 0)));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() >= 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

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

    if (cropAxis == 0) {
        VPU_THROW_EXCEPTION
            << "Layer " << layer->name << " [" << layer->type
            << "] Can't crop batch channel";
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
