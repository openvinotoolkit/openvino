// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <string>

namespace vpu {

using InferenceEngine::CNNLayerPtr;

//----------------------------------------------------------------------

namespace {

class ScatterElementsUpdateStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ScatterElementsUpdateStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto data = inputEdge(0)->input();
        const auto indices = inputEdge(1)->input();
        const auto updates = inputEdge(2)->input();
        const auto axis = inputEdge(3)->input();
        const auto output = outputEdge(0)->output();
        orderInfo.setInput(inputEdge(0), DimsOrder::fromNumDims(data->desc().numDims()));
        orderInfo.setInput(inputEdge(1), DimsOrder::fromNumDims(indices->desc().numDims()));
        orderInfo.setInput(inputEdge(2), DimsOrder::fromNumDims(updates->desc().numDims()));
        orderInfo.setInput(inputEdge(3), DimsOrder::fromNumDims(axis->desc().numDims()));
        orderInfo.setOutput(outputEdge(0), DimsOrder::fromNumDims(output->desc().numDims()));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());    // `data`    tensor
        stridesInfo.setInput(inputEdge(1), StridesRequirement::compact());    // `indices` tensor
        stridesInfo.setInput(inputEdge(2), StridesRequirement::compact());    // `updates` tensor
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());  // `output`  tensor
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        const auto& srcType = input(0)->desc().type();
        assertInputsOutputsTypes(this, {{srcType}, {DataType::S32}, {srcType}, {DataType::S32}}, {{srcType}});
        //                               `data`  ,  `indices`     , `updates`,  `axis`         ,   `output`
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto data    = input(0);
        auto indices = input(1);
        auto updates = input(2);
        auto axis    = input(3);
        auto out = output(0);

        data->serializeBuffer(serializer);
        out->serializeBuffer(serializer);
        indices->serializeBuffer(serializer);
        updates->serializeBuffer(serializer);
        axis->serializeBuffer(serializer);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
    }
};

}  // namespace

//----------------------------------------------------------------------

static
void checkTensorShapes(const vpu::Data& input,
                       const vpu::Data& output,
                       const vpu::Data& indices,
                       const vpu::Data& updates,
                       const vpu::Data& axis) {
    const DataDesc& inputDesc = input->desc();
    const DataDesc& outputDesc = output->desc();
    const DataDesc& indicesDesc = indices->desc();
    const DataDesc& updatesDesc = updates->desc();
    const DataDesc& axisDesc = axis->desc();

    const auto inputType = inputDesc.type();
    const auto outputType = outputDesc.type();
    const auto indicesType = indicesDesc.type();
    const auto updatesType = updatesDesc.type();
    const auto axisType = axisDesc.type();

    VPU_THROW_UNLESS(inputType == DataType::S32 ||
                     inputType == DataType::FP16, "input type is invalid");
    VPU_THROW_UNLESS(outputType == inputType, "output type is invalid");
    VPU_THROW_UNLESS(updatesType == inputType, "updates type is invalid");
    VPU_THROW_UNLESS(indicesType == DataType::S32, "indices type is invalid");
    VPU_THROW_UNLESS(axisType == DataType::S32, "axis type is invalid");

    const int inputNDims = inputDesc.numDims();
    const int outputNDims = outputDesc.numDims();
    const int indicesNDims = indicesDesc.numDims();
    const int updatesNDims = updatesDesc.numDims();
    const int axisNDims = axisDesc.numDims();

    VPU_THROW_UNLESS(inputNDims > 0, "input tensor must not be 0-dimensional");
    VPU_THROW_UNLESS(outputNDims > 0, "output tensor must not be 0-dimensional");
    VPU_THROW_UNLESS(indicesNDims > 0, "indices tensor must not be 0-dimensional");
    VPU_THROW_UNLESS(updatesNDims > 0, "updates tensor must not be 0-dimensional");
    VPU_THROW_UNLESS(axisNDims > 0, "axis tensor must not be 0-dimensional");

    VPU_THROW_UNLESS(inputNDims == outputNDims,
                     "input and output have different shapes: inputNDims={}, outputNDims={}",
                     inputNDims, outputNDims);

    VPU_THROW_UNLESS(inputNDims == indicesNDims,
                     "input and indices have different shapes: inputNDims={}, indicesNDims={}",
                     inputNDims, updatesNDims);

    VPU_THROW_UNLESS(inputNDims == updatesNDims,
                     "input and updates have different shapes: inputNDims={}, updatesNDims={}",
                     inputNDims, updatesNDims);

    VPU_THROW_UNLESS(axisNDims == 1,
                     "axis tensor must be 1-dimensional, but axisNDims={}",
                     axisNDims);

    const DimsOrder inputDimsOrder = inputDesc.dimsOrder();
    const DimsOrder outputDimsOrder = outputDesc.dimsOrder();
    const DimsOrder indicesDimsOrder = indicesDesc.dimsOrder();
    const DimsOrder updatesDimsOrder = updatesDesc.dimsOrder();
    const DimsOrder axisDimsOrder = axisDesc.dimsOrder();

    VPU_THROW_UNLESS(inputDimsOrder == outputDimsOrder, "output must have same layout as input"
                     ", but inputDimsOrder = \"{}\", and outputDimsOrder = \"{}\"",
                     inputDimsOrder, outputDimsOrder);

    VPU_THROW_UNLESS(inputDimsOrder == indicesDimsOrder, "indices must have same layout as input"
                     ", but inputDimsOrder = \"{}\", and indicesDimsOrder = \"{}\"",
                     inputDimsOrder, indicesDimsOrder);

    VPU_THROW_UNLESS(inputDimsOrder == updatesDimsOrder, "updates must have same layout as input"
                     ", but inputDimsOrder = \"{}\", and updatesDimsOrder = \"{}\"",
                     inputDimsOrder, updatesDimsOrder);

    const DimValues& inputDims = inputDesc.dims();
    const DimValues& outputDims = outputDesc.dims();
    const DimValues& indicesDims = indicesDesc.dims();
    const DimValues& updatesDims = updatesDesc.dims();
    const DimValues& axisDims = axisDesc.dims();

    VPU_THROW_UNLESS(inputDims == outputDims, "input and output tensors must have same lengths"
                     ", but inputDims = \"{}\", and outputDims = \"{}\"", inputDims, outputDims);

    VPU_THROW_UNLESS(indicesDims == updatesDims, "indices and updates tensors must have same lengths"
                     ", but indicesDims = \"{}\", and updatesDims = \"{}\"", indicesDims, updatesDims);

    // Permutation is array of dims, from minor to major
    const DimVector outputPerm = outputDimsOrder.toPermutation();
    const DimVector updatesPerm = updatesDimsOrder.toPermutation();

    // Check if the updates fits the data shape
    for (int i = 0; i < inputNDims - 1; i++) {
        const Dim outputDim = outputPerm[i];
        const Dim updatesDim = updatesPerm[i];
        const int outputSize = outputDims[outputDim];
        const int updatesSize = updatesDims[updatesDim];
        VPU_THROW_UNLESS(updatesSize <= outputSize,
                         "updates size must fit output for corresponding axes, "
                         "but for axis={}: output size={}, updates size={}",
                         i, outputSize, updatesSize);
    }

    // Note, that for a 1D tensor the layout is "C"
    VPU_THROW_UNLESS(axisDimsOrder == DimsOrder::C,
                     "axis must be 1D tensor, but its dims order is {}",
                     axisDimsOrder);
    VPU_THROW_UNLESS(axisDims[Dim::C] == 1,
                     "axis tensor must be 1D array of 1 element, but axis length = %d",
                     axisDims[Dim::C]);
}

void FrontEnd::parseScatterElementsUpdate(const Model      & model,
                                          const CNNLayerPtr& layer,
                                          const DataVector & inputs,
                                          const DataVector & outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 4, "invalid number of inputs: %lu", inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1, "invalid number of outputs: %lu", outputs.size());

    const auto& input   = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    const auto& axis    = inputs[3];
    const auto& output = outputs[0];

    checkTensorShapes(input, output, indices, updates, axis);

    auto scatterElementsUpdateLayer = std::dynamic_pointer_cast<ie::ScatterElementsUpdateLayer>(layer);

    VPU_THROW_UNLESS(scatterElementsUpdateLayer != nullptr,
                     "this layer is not an instance of ScatterElementsUpdateLayer: "
                     "layer name = \"%s\", layer type = \"%s\"",
                     layer->name.c_str(), layer->type.c_str());

    auto stage = model->addNewStage<ScatterElementsUpdateStage>(layer->name,
                                                                StageType::ScatterElementsUpdate,
                                                                layer,
                                                                {input, indices, updates, axis},
                                                                {output});

    VPU_THROW_UNLESS(stage != nullptr,
                     "failed to create ScatterElementsUpdateStage: "
                     "layer name = \"%s\", layer type = \"%s\"",
                     layer->name.c_str(), layer->type.c_str());
}

//----------------------------------------------------------------------

Stage StageBuilder::addScatterElementsUpdateStage(const Model& model,
                                                  const std::string& name,
                                                  const ie::CNNLayerPtr& layer,
                                                  const Data& input,
                                                  const Data& output,
                                                  const Data& indices,
                                                  const Data& updates,
                                                  const Data& axis) {
    checkTensorShapes(input, output, indices, updates, axis);

    auto stage = model->addNewStage<ScatterElementsUpdateStage>(name,
                                                                StageType::ScatterElementsUpdate,
                                                                layer,
                                                                {input, indices, updates, axis},
                                                                {output});

    VPU_THROW_UNLESS(stage != nullptr,
                     "failed to create ScatterElementsUpdateStage: "
                     "layer name = \"%s\", layer type = \"%s\"",
                     layer->name.c_str(), layer->type.c_str());

    return stage;
}

}  // namespace vpu
