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
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<LSTMCellStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 5);
        IE_ASSERT(_outputEdges.size() == 2);

        auto output = _outputEdges[0]->output();
        auto input = _inputEdges[0]->input();

        auto inputDimsOrder = input->desc().dimsOrder();
        auto outputDimsOrder = output->desc().dimsOrder();

        if (inputDimsOrder.numDims() >= 3) {
            inputDimsOrder.moveDim(Dim::C, 2);  // ->...CHW
        }
        if (outputDimsOrder.numDims() >= 3) {
            outputDimsOrder.moveDim(Dim::C, 2);  // ->...CHW
        }

        _orderInfo.setInput(_inputEdges[0], inputDimsOrder);
        _orderInfo.setOutput(_outputEdges[0], outputDimsOrder);
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 5);
        IE_ASSERT(_outputEdges.size() == 2);

        for (const auto& inEdge : _inputEdges) {
            _stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : _outputEdges) {
            _stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    void finalCheckImpl() const override {
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
        IE_ASSERT(_inputEdges.size() == 5);
        IE_ASSERT(_outputEdges.size() == 2);

        int nCells = attrs().get<int>("nCells");

        bool useTempBuffer = (nCells > 1);
        IE_ASSERT((_tempBufferEdges.size() == 1 && useTempBuffer) || !useTempBuffer);

        for (const auto& inEdge : _inputEdges) {
            inEdge->input()->serializeNewBuffer(serializer);
        }
        for (const auto& outEdge : _outputEdges) {
            outEdge->output()->serializeNewBuffer(serializer);
        }

        if (useTempBuffer)
            _tempBufferEdges[0]->tempBuffer()->serializeNewBuffer(serializer);
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

    size_t state_size = inputs[1]->desc().totalDimSize() / nBatches;
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

    Data weights, biases;
    std::tie(weights, biases) = getWeightsAndBiases(model, layer);

    auto stage = model->addNewStage<LSTMCellStage>(
            layer->name,
            StageType::LSTMCell,
            layer,
            {inputs[0], inputs[1], inputs[2], weights, biases},
            outputs);
    stage->attrs().set<bool>("RNNForward", true);
    stage->attrs().set<int>("nCells", 1);
    stage->attrs().set<int>("nBatches", 1);
}

}  // namespace vpu
