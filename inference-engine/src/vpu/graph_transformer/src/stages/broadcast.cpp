// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/utils/numeric.hpp>

#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpu {

namespace {

class BroadcastStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<BroadcastStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto inputOrder = input(0)->desc().dimsOrder();
        auto outputOrder = DimsOrder::fromNumDims(output(0)->desc().numDims());

        if (inputOrder.numDims() >= 3 && inputOrder.dimInd(Dim::C) == 0) {
            outputOrder.moveDim(Dim::C, 0);
        }

        orderInfo.setOutput(outputEdge(0), outputOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement().remove(0));
        stridesInfo.setOutput(outputEdge(0), StridesRequirement().remove(0));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        const auto mode = attrs().getOrDefault<BroadcastMode>("mode", BroadcastMode::NUMPY);
        const auto& dataPrecision = input(0)->desc().type();

        VPU_THROW_UNLESS(numOutputs() == 1,
                         "{} stage with name {} must have only 1 output, actually provided {} outputs",
                         type(), name(), numOutputs());
        if (mode == BroadcastMode::NUMPY) {
            VPU_THROW_UNLESS(numInputs() == 2,
                             "{} stage with name {} and numpy mode must have 2 inputs, actually "
                             "provided {} inputs", type(), name(), numInputs());
            assertInputsOutputsTypes(this,
                                     {{dataPrecision}, {DataType::S32}},
                                     {{dataPrecision}});

        } else {
            VPU_THROW_UNLESS(numInputs() == 3,
                             "{} stage with name {} and explicit mode must have 3 inputs, actually "
                             "provided {} inputs", type(), name(), numInputs());
            assertInputsOutputsTypes(this,
                                     {{dataPrecision}, {DataType::S32}, {DataType::S32}},
                                     {{dataPrecision}});
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto mode = attrs().getOrDefault<BroadcastMode>("mode", BroadcastMode::NUMPY);
        serializer.append(static_cast<uint32_t>(mode == BroadcastMode::NUMPY ? 0 : 1));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        const auto mode = attrs().getOrDefault<BroadcastMode>("mode", BroadcastMode::NUMPY);

        input(0)->serializeBuffer(serializer);
        input(1)->serializeBuffer(serializer);
        if (mode == BroadcastMode::EXPLICIT) {
            input(2)->serializeBuffer(serializer);
        }
        output(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseBroadcast(
        const Model& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) const {
    VPU_THROW_UNLESS(layer != nullptr,
                     "parseBroadcast expects valid CNNLayerPtr, got nullptr");

    VPU_THROW_UNLESS(outputs.size() == 1,
                     "{} layer with name {} must have only 1 output, actually provided {} outputs",
                     layer->type, layer->name, outputs.size());
    const auto output = outputs[0];

    const auto modeString = layer->GetParamAsString("mode", "numpy");
    if (modeString == "numpy") {
        VPU_THROW_UNLESS(inputs.size() == 2,
                         "{} layer with name {} and numpy mode must have 2 inputs, actually "
                         "provided {} inputs", layer->type, layer->name, inputs.size());
    } else if (modeString == "explicit") {
        VPU_THROW_UNLESS(inputs.size() == 3,
                         "{} layer with name {} and explicit mode must have 3 inputs, actually "
                         "provided {} inputs", layer->type, layer->name, inputs.size());
        const auto axesMappingDesc = inputs[2]->desc();
        const auto axesMappingPerm = axesMappingDesc.dimsOrder().toPermutation();
        const auto axesMappingDim = axesMappingDesc.dim(axesMappingPerm.at(0));
        VPU_THROW_UNLESS(axesMappingDesc.numDims() == 1,
                         "{} layer with name {} and explicit mode must have 1D axesMapping tensor, "
                         "actually provided {}D tensor",
                         layer->type, layer->name, axesMappingDesc.numDims());
        VPU_THROW_UNLESS(axesMappingDim == inputs[0]->desc().numDims(),
                         "{} layer with name {} and explicit mode must have axesMapping tensor with "
                         "size equals to number of output dims, expected [{}], provided [{}]",
                         layer->type, layer->name, output->desc().numDims(), axesMappingDim);

    } else {
        VPU_THROW_FORMAT("{} layer with name {}: Graph Transformer doesn't support {} mode",
                         layer->type, layer->name, modeString);
    }

    const auto shape = inputs[1];
    const auto shapeDesc = inputs[1]->desc();
    const auto shapeDim = shapeDesc.dim(shapeDesc.dimsOrder().toPermutation().at(0));
    VPU_THROW_UNLESS(shapeDesc.numDims() == 1,
                     "{} layer with name {} and explicit mode must have 1D target shape tensor, "
                     "actually provided {}D tensor",
                     layer->type, layer->name, shapeDesc.numDims());
    VPU_THROW_UNLESS(shapeDim == output->desc().numDims(),
                     "{} layer with name {} and explicit mode must have target shape tensor with "
                     "size equals to number of output dims, expected [{}], provided [{}]",
                     layer->type, layer->name, output->desc().numDims(), shapeDim);

    const auto mode = modeString == "numpy" ? BroadcastMode::NUMPY : BroadcastMode::EXPLICIT;

    auto stage = model->addNewStage<BroadcastStage>(
            layer->name,
            StageType::Broadcast,
            layer,
            inputs,
            outputs);

    stage->attrs().set("mode", mode);
}

}  //namespace vpu
