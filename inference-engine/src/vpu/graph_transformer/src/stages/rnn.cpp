// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <memory>
#include <set>

#include <vpu/utils/numeric.hpp>

namespace vpu {

namespace {

class LSTMCellStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<LSTMCellStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto output = outputEdge(0)->output();
        auto input = inputEdge(0)->input();

        auto inputDimsOrder = input->desc().dimsOrder();
        auto outputDimsOrder = output->desc().dimsOrder();

        if (inputDimsOrder.numDims() >= 3) {
            inputDimsOrder.moveDim(Dim::C, 2);  // ->...CHW
        }
        if (outputDimsOrder.numDims() >= 3) {
            outputDimsOrder.moveDim(Dim::C, 2);  // ->...CHW
        }

        orderInfo.setInput(inputEdge(0), inputDimsOrder);
        orderInfo.setOutput(outputEdge(0), outputDimsOrder);
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
        IE_ASSERT(numInputs() == 5);
        IE_ASSERT(numOutputs() > 0);
        assertAllInputsOutputsTypes(this, DataType::FP16, DataType::FP16);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto RNNForward = attrs().get<bool>("RNNForward");
        auto nCells = attrs().get<int>("nCells");
        auto nBatches = attrs().get<int>("nBatches");
        serializer.append(static_cast<int>(RNNForward));
        serializer.append(static_cast<int>(nCells));
        serializer.append(static_cast<int>(nBatches));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        int nCells = attrs().get<int>("nCells");

        bool useTempBuffer = (nCells > 1);
        IE_ASSERT((numTempBuffers() == 1 && useTempBuffer) || !useTempBuffer);

        for (const auto& inEdge : inputEdges()) {
            inEdge->input()->serializeNewBuffer(serializer);
        }
        for (const auto& outEdge : outputEdges()) {
            outEdge->output()->serializeNewBuffer(serializer);
        }

        if (useTempBuffer) {
            tempBuffer(0)->serializeNewBuffer(serializer);
        }
    }
};

}  // namespace

static void RNNRelayout(
                 const fp16_t* src,
                 fp16_t* dst0,
                 fp16_t* dst1,

                 const int ngates,
                 const int state_size,
                 const int input_size
                ) {
    int counter = 0;
    for (int j = 0; j < ngates * state_size; j++) {
        for (int i = 0; i < input_size; i++) {
            dst0[(input_size) * j + i] = src[counter++];
        }
        for (int i = 0; i < state_size; i++) {
            dst1[(state_size) * j + i] = src[counter++];
        }
    }
}

void FrontEnd::parseRNN(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector &inputs,
        const DataVector &outputs) {
    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::RNNSequenceLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    const int ngates = 4;

    Data weights, biases;
    std::tie(weights, biases) = getWeightsAndBiases(model, layer);

    size_t nCells = inputs[0]->desc().dim(Dim::H);
    size_t nBatches = inputs[0]->desc().dim(Dim::C);
    IE_ASSERT(nCells >= 1);
    IE_ASSERT(nBatches >= 1);

    size_t input_size = inputs[0]->desc().dim(Dim::W);
    IE_ASSERT(input_size == inputs[0]->desc().totalDimSize() / nCells / nBatches);

    size_t state_size = outputs[0]->desc().totalDimSize() / nCells / nBatches;
    size_t cell_state_size = inputs[2]->desc().totalDimSize() / nBatches;
    IE_ASSERT(state_size == cell_state_size);

    size_t weightsSize = weights->desc().totalDimSize();
    IE_ASSERT(state_size * (input_size + state_size) * ngates == weightsSize);

    size_t biasesSize = biases->desc().totalDimSize();
    IE_ASSERT(state_size * ngates == biasesSize);

    /* weights repacking */
    auto newWeightsBlob = ie::make_shared_blob<fp16_t>(ie::TensorDesc(
        ie::Precision::FP16,
        {weightsSize},
        ie::Layout::C));
    newWeightsBlob->allocate();
    auto newWeightsPtr = newWeightsBlob->buffer().as<fp16_t*>();
    auto content = weights->content();
    IE_ASSERT(content != nullptr);
    auto origWeights = content->get<fp16_t>();
    IE_ASSERT(origWeights != nullptr);
    RNNRelayout(origWeights,
                newWeightsPtr,
                newWeightsPtr + ngates * state_size * input_size,

                ngates,
                state_size,
                input_size);

    auto newWeights = model->addConstData(
        _layer->name + "@weights",
        weights->desc(),
        ieBlobContent(newWeightsBlob));

    auto stateCellFinal = model->addFakeData();
    auto stage = model->addNewStage<LSTMCellStage>(
        layer->name,
        StageType::LSTMCell,
        layer,
        {inputs[0], inputs[1], inputs[2], newWeights, biases},
        {outputs[0], stateCellFinal});

    if (nCells > 1)
        model->addTempBuffer(stage, DataDesc({state_size}));

    bool RNNForward = layer->direction == ie::RNNSequenceLayer::FWD;
    stage->attrs().set<bool>("RNNForward", RNNForward);
    stage->attrs().set<int>("nCells", nCells);
    stage->attrs().set<int>("nBatches", nBatches);
}

void FrontEnd::parseLSTMCell(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector &inputs,
        const DataVector &outputs) {
    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() == 2);

    auto layer = std::dynamic_pointer_cast<ie::LSTMCell>(_layer);
    IE_ASSERT(layer != nullptr);

    DataVector stageInputs = inputs;
    auto origWeights = layer->_weights;

    IE_ASSERT(origWeights != nullptr) << "weights are empty for layer: " << layer->name;

    if (lstmWeights.count(origWeights) != 0) {
        stageInputs.emplace_back(lstmWeights[origWeights]);
    } else {
        auto weights = model->addConstData(
                layer->name + "@weights",
                DataDesc({origWeights->size()}),
                ieBlobContent(origWeights));
        lstmWeights[origWeights] = weights;
        stageInputs.emplace_back(weights);
    }

    auto origBiases = layer->_biases;

    Data biases;
    if (origBiases == nullptr) {
        biases = model->addFakeData();
    } else {
        if (lstmBiases.count(origBiases) != 0) {
            biases = lstmBiases[origBiases];
        } else {
            biases = model->addConstData(
                    layer->name + "@biases",
                    DataDesc({origBiases->size()}),
                    ieBlobContent(origBiases));
            lstmBiases[origBiases] = biases;
        }
    }

    stageInputs.emplace_back(biases);

    auto stage = model->addNewStage<LSTMCellStage>(
            layer->name,
            StageType::LSTMCell,
            layer,
            stageInputs,
            outputs);
    stage->attrs().set<bool>("RNNForward", true);
    stage->attrs().set<int>("nCells", 1);
    stage->attrs().set<int>("nBatches", 1);
}

}  // namespace vpu
