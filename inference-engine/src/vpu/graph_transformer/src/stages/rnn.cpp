// Copyright (C) 2018-2021 Intel Corporation
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

        VPU_THROW_UNLESS(inputEdges().size() == 5,
                         "LSTMCell: input edges: {}, but expected: 5",
                         inputEdges().size());

        // check number of outputs, without temp buffer
        const int outputsNumber = static_cast<int>(outputEdges().size());
        const int useCellState = outputsNumber >= 2;
        const int outputEdgesExpected = 1
                                      + (useCellState ? 1 : 0)
                                      + (outputsNumber == 3 ? 1 : 0);
        VPU_THROW_UNLESS(outputEdges().size() == outputEdgesExpected,
                         "LSTMCell: number of output edges: {}, but expected: {}",
                         outputEdges().size(), outputEdgesExpected);

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

static int64_t get_seq_axis(const std::shared_ptr<ngraph::Node>& sequence_node) {
    // Optimization.
    // Plug-ins support seq_axis attribute (value 1 or 0) for Seq ops, but according to the spec we don't
    // support this attribute and should insert Transpose layer before and after Seq op in TI to Sequences
    // transformation. Additional Transpose layers affect the performance, so we try to detect pattern
    // Transpose(axis_order={1,0,2}) -> Seq -> Transpose(axis_order={2,1,0,3}
    // and replace unnecessary Transpose ops with SeqIE (seq_axis = 0) to transfer value
    // of the attribute to plug-ins.
    // todo: specify seq_axis attribute for Sequence ops.
    int64_t seq_axis = 1; // default
    const auto& target_inputs = sequence_node->output(0).get_target_inputs();
    if (target_inputs.size() == 1) {
        const auto& transpose_before = std::dynamic_pointer_cast<ngraph::opset5::Transpose>(sequence_node->input_value(0).get_node_shared_ptr());
        const auto& transpose_after = std::dynamic_pointer_cast<ngraph::opset5::Transpose>(target_inputs.begin()->get_node()->shared_from_this());
        if (transpose_after != nullptr && transpose_before != nullptr) {
            auto order_before = std::dynamic_pointer_cast<ngraph::opset5::Constant>(
                    transpose_before->input_value(1).get_node_shared_ptr());
            auto order_after = std::dynamic_pointer_cast<ngraph::opset5::Constant>(
                    transpose_after->input_value(1).get_node_shared_ptr());
            if (order_before != nullptr && order_after != nullptr) {
                auto order_before_values = order_before->cast_vector<int64_t>();
                auto order_after_values = order_after->cast_vector<int64_t>();
                std::vector<int64_t> order_ref_before = {1, 0, 2};
                std::vector<int64_t> order_ref_after = {2, 1, 0, 3};
                if (order_before_values == order_ref_before && order_after_values == order_ref_after) {
                    seq_axis = 0;
                }
            }
        }
    }
    return seq_axis;
}

void FrontEnd::parseRNN(const Model& model, const NodePtr& node, const DataVector &inputs, const DataVector &outputs) const {
    const auto& rnnSequence = ngraph::as_type_ptr<ngraph::op::v5::RNNSequence>(node);
    IE_ASSERT(rnnSequence != nullptr);
    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() <= 3);

    const int ngates = 4;

    Data weights, biases;
    const auto weightsNode = node->input_value(4).get_node_shared_ptr();
    const auto biasNode = node->input_value(5).get_node_shared_ptr();
    std::tie(weights, biases) = getWeightsAndBiases(model, rnnSequence->get_friendly_name(), weightsNode, biasNode);

    size_t nCells = 0;
    size_t nBatches = 0;
    size_t inputSize = inputs[0]->desc().dim(Dim::W);
    auto axis = get_seq_axis(rnnSequence);
    if (axis == 1) {
        nCells = inputs[0]->desc().dim(Dim::H);
        nBatches = inputs[0]->desc().dim(Dim::C);
    } else if (axis == 0) {
        nCells = inputs[0]->desc().dim(Dim::C);
        nBatches = inputs[0]->desc().dim(Dim::H);
    } else if (axis == 2) {
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
            static_cast<int>(stateSize),
            static_cast<int>(inputSize));
    };

    auto newWeights = model->addConstData(rnnSequence->get_friendly_name() + "@weights", weights->desc(), generator);
    auto outputData = outputs;

    const auto& validOutDataIter = std::find_if(outputData.begin(), outputData.end(), [](const vpu::Data& data) {
        return data != nullptr;
    });

    VPU_THROW_UNLESS(validOutDataIter != outputData.end(), "Layer {} with type {} failed: all outputs is nullptr", rnnSequence->get_friendly_name(), rnnSequence->get_type_name());

    for (auto& out : outputData) {
        if (out == nullptr) {
            out = model->addFakeData();
        }
    }

    auto stage = model->addNewStage<LSTMCellStage>(
        rnnSequence->get_friendly_name(),
        StageType::LSTMCell,
        rnnSequence,
        {inputs[0], inputs[1], inputs[2], newWeights, biases},
        outputData);

    if (nCells > 1)
        model->addTempBuffer(stage, sizeof(uint16_t) * stateSize);
    bool RNNForward = rnnSequence->get_direction() == ngraph::op::RecurrentSequenceDirection::FORWARD;
    stage->attrs().set<bool>("RNNForward", RNNForward);
    stage->attrs().set<int>("nCells", static_cast<int>(nCells));
    stage->attrs().set<int>("nBatches", static_cast<int>(nBatches));
}

void FrontEnd::parseLSTMCell(const Model& model, const NodePtr& node, const DataVector &inputs, const DataVector &outputs) {
    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() == 2);

    const int ngates = 4;

    const auto lstmCell = ngraph::as_type_ptr<ngraph::opset4::LSTMCell>(node);
    VPU_THROW_UNLESS(lstmCell != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());

    Data weights;
    Data biases;
    std::tie(weights, std::ignore) = getWeightsAndBiases(model,
                                                         lstmCell->get_friendly_name(),
                                                         node->input_value(3).get_node_shared_ptr(),
                                                         node->input_value(4).get_node_shared_ptr());

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
            static_cast<int>(stateSize),
            static_cast<int>(inputSize));
    };

    auto newWeights = model->addConstData(lstmCell->get_name() + "@weights", weights->desc(), generator);

    DataVector stageInputs = inputs;
    auto origWeights = shareWeights(node->input_value(3).get_node_shared_ptr());

    IE_ASSERT(origWeights != nullptr) << "weights are empty for node: " << lstmCell->get_friendly_name();

    if (_lstmWeights.count(origWeights) != 0) {
        stageInputs.emplace_back(_lstmWeights[origWeights]);
    } else {
        _lstmWeights[origWeights] = newWeights;
        stageInputs.emplace_back(newWeights);
    }

    auto origBiases = shareWeights(node->input_value(4).get_node_shared_ptr());

    
    if (origBiases == nullptr) {
        biases = model->addFakeData();
    } else {
        if (_lstmBiases.count(origBiases) != 0) {
            biases = _lstmBiases[origBiases];
        } else {
            biases = model->addConstData(
                    lstmCell->get_friendly_name() + "@biases",
                    DataDesc({origBiases->size()}),
                    ieBlobContent(origBiases));
            _lstmBiases[origBiases] = biases;
        }
    }

    stageInputs.emplace_back(biases);

    // Filter out handles to data which are nullptr (they are excluded in FrontEnd::getInputAndOutputData)
    DataVector realOutputs;
    std::copy_if(outputs.cbegin(), outputs.cend(), std::back_inserter(realOutputs), [](const Data& handle) { return !!handle;});

    auto stage = model->addNewStage<LSTMCellStage>(lstmCell->get_friendly_name(), StageType::LSTMCell, lstmCell, stageInputs, realOutputs);
    stage->attrs().set<bool>("RNNForward", true);
    stage->attrs().set<int>("nCells", 1);
    stage->attrs().set<int>("nBatches", static_cast<int>(nBatches));
}

}  // namespace vpu
