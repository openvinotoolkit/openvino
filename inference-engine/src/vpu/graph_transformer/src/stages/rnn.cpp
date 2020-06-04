// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/utils/numeric.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <vector>
#include <string>
#include <memory>
#include <set>

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
        IE_ASSERT(numOutputs() < 4);
        assertAllInputsOutputsTypes(this, DataType::FP16, DataType::FP16);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto RNNForward = attrs().get<bool>("RNNForward");
        const auto nCells     = attrs().get<int>("nCells");
        const auto nBatches   = attrs().get<int>("nBatches");
        const bool useCellState = outputEdges().size() >= 2;
        serializer.append(static_cast<int>(RNNForward));
        serializer.append(nCells);
        serializer.append(nBatches);
        serializer.append(static_cast<int>(useCellState));
        serializer.append(static_cast<int>(outputEdges().size()));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        int nCells = attrs().get<int>("nCells");

        bool useTempBuffer = (nCells > 1);
        IE_ASSERT((numTempBuffers() == 1 && useTempBuffer) || !useTempBuffer);

        for (const auto& inEdge : inputEdges()) {
            inEdge->input()->serializeBuffer(serializer);
        }
        for (const auto& outEdge : outputEdges()) {
            outEdge->output()->serializeBuffer(serializer);
        }

        if (useTempBuffer) {
            tempBuffer(0)->serializeBuffer(serializer);
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

void FrontEnd::parseRNN(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector &inputs, const DataVector &outputs) const {
    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() <= 3);

    auto layer = std::dynamic_pointer_cast<ie::RNNSequenceLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    const int ngates = 4;

    Data weights, biases;
    std::tie(weights, biases) = getWeightsAndBiases(model, layer);

    size_t nCells = 0;
    size_t nBatches = 0;
    size_t inputSize = inputs[0]->desc().dim(Dim::W);

    if (layer->axis == 1) {
        nCells = inputs[0]->desc().dim(Dim::H);
        nBatches = inputs[0]->desc().dim(Dim::C);
    } else if (layer->axis == 0) {
        nCells = inputs[0]->desc().dim(Dim::C);
        nBatches = inputs[0]->desc().dim(Dim::H);
    } else if (layer->axis == 2) {
        nCells = inputs[0]->desc().dim(Dim::W);
        nBatches = inputs[0]->desc().dim(Dim::C);
        inputSize = inputs[0]->desc().dim(Dim::H);
    }

    IE_ASSERT(nCells >= 1);
    IE_ASSERT(nBatches >= 1);

    IE_ASSERT(inputSize == inputs[0]->desc().totalDimSize() / nCells / nBatches);

    size_t stateSize = outputs[0]->desc().totalDimSize() / nCells / nBatches;
    size_t cellStateSize = inputs[2]->desc().totalDimSize() / nBatches;

    IE_ASSERT(stateSize == cellStateSize);

    size_t weightsSize = weights->desc().totalDimSize();
    IE_ASSERT(stateSize * (inputSize + stateSize) * ngates == weightsSize);

    size_t biasesSize = biases->desc().totalDimSize();
    IE_ASSERT(stateSize * ngates == biasesSize);

    /* weights repacking */
    const auto generator = [&weights, stateSize, inputSize, ngates, outputs](const ie::Blob::Ptr& blob) {
        auto newWeightsPtr = blob->buffer().as<fp16_t*>();

        auto content = weights->content();
        IE_ASSERT(content != nullptr);

        auto origWeights = content->get<fp16_t>();
        IE_ASSERT(origWeights != nullptr);

        RNNRelayout(
            origWeights,

            newWeightsPtr,
            newWeightsPtr + ngates * stateSize * inputSize,

            ngates,
            stateSize,
            inputSize);
    };

    auto newWeights = model->addConstData(_layer->name + "@weights", weights->desc(), generator);
    auto stateCellFinal = model->addFakeData();
    auto outputKeeper = {outputs[0], stateCellFinal};

    if (outputs.size() == 2) {
        outputKeeper = {outputs[0], outputs[1]};
    } else if (outputs.size() == 3) {
        outputKeeper = {outputs[0], outputs[1], outputs[2]};
    }

    auto stage = model->addNewStage<LSTMCellStage>(
        layer->name,
        StageType::LSTMCell,
        layer,
        {inputs[0], inputs[1], inputs[2], newWeights, biases},
        outputKeeper);

    if (nCells > 1)
        model->addTempBuffer(stage, DataDesc({stateSize}));

    bool RNNForward = layer->direction == ie::RNNSequenceLayer::FWD;
    stage->attrs().set<bool>("RNNForward", RNNForward);
    stage->attrs().set<int>("nCells", nCells);
    stage->attrs().set<int>("nBatches", nBatches);
}

void FrontEnd::parseLSTMCell(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector &inputs, const DataVector &outputs) {
    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() <= 3);

    const int ngates = 4;

    const auto layer = std::dynamic_pointer_cast<ie::LSTMCell>(_layer);
    IE_ASSERT(layer != nullptr);

    Data weights;
    std::tie(weights, std::ignore) = getWeightsAndBiases(model, layer);

    const auto& src = inputs[0]->desc();

    const std::size_t nBatches = src.dim(Dim::N);

    IE_ASSERT(nBatches >= 1);

    const std::size_t stateSize = outputs[0]->desc().totalDimSize() / nBatches;

    IE_ASSERT(src.numDims() == 2);
    const std::size_t inputSize = src.dim(Dim::C);

    const std::size_t cellStateSize = inputs[2]->desc().totalDimSize() / nBatches;

    IE_ASSERT(inputSize == src.totalDimSize() / nBatches);

    IE_ASSERT(stateSize == cellStateSize);

    const std::size_t weightsSize = weights->desc().totalDimSize();

    IE_ASSERT(stateSize * (inputSize + stateSize) * ngates == weightsSize);

    const auto generator = [&weights, stateSize, inputSize, ngates](const ie::Blob::Ptr& blob) {
        auto newWeightsPtr = blob->buffer().as<fp16_t*>();

        auto content = weights->content();
        IE_ASSERT(content != nullptr);

        auto origWeights = content->get<fp16_t>();
        IE_ASSERT(origWeights != nullptr);

        RNNRelayout(
            origWeights,

            newWeightsPtr,
            newWeightsPtr + ngates * stateSize * inputSize,

            ngates,
            stateSize,
            inputSize);
    };

    auto newWeights = model->addConstData(_layer->name + "@weights", weights->desc(), generator);
    DataVector stageInputs = inputs;
    auto origWeights = layer->_weights;

    IE_ASSERT(origWeights != nullptr) << "weights are empty for layer: " << layer->name;

    if (_lstmWeights.count(origWeights) != 0) {
        stageInputs.emplace_back(_lstmWeights[origWeights]);
    } else {
        _lstmWeights[origWeights] = newWeights;
        stageInputs.emplace_back(newWeights);
    }

    auto origBiases = layer->_biases;

    Data biases;
    if (origBiases == nullptr) {
        biases = model->addFakeData();
    } else {
        if (_lstmBiases.count(origBiases) != 0) {
            biases = _lstmBiases[origBiases];
        } else {
            biases = model->addConstData(
                    layer->name + "@biases",
                    DataDesc({origBiases->size()}),
                    ieBlobContent(origBiases));
            _lstmBiases[origBiases] = biases;
        }
    }

    stageInputs.emplace_back(biases);

    // Filter out handles to data which are nullptr (they are excluded in FrontEnd::getInputAndOutputData)
    DataVector realOutputs;
    std::copy_if(outputs.cbegin(), outputs.cend(), std::back_inserter(realOutputs), [](const Data& handle) { return !!handle;});

    auto stage = model->addNewStage<LSTMCellStage>(layer->name, StageType::LSTMCell, layer, stageInputs, realOutputs);
    stage->attrs().set<bool>("RNNForward", true);
    stage->attrs().set<int>("nCells", 1);
    stage->attrs().set<int>("nBatches", nBatches);
}

}  // namespace vpu
