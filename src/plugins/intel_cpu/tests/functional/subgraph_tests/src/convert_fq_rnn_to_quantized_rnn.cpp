// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/runtime/tensor.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

using namespace CPUTestUtils;

namespace ov {
namespace test {

using ConvertFqRnnToQuantizedRnnTestParams = std::tuple<std::string, std::vector<InputShape>, bool>;

class ConvertFqRnnToQuantizedRnn : public testing::WithParamInterface<ConvertFqRnnToQuantizedRnnTestParams>,
                                   public CpuTestWithFusing,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertFqRnnToQuantizedRnnTestParams>& obj) {
        std::vector<InputShape> inputShapes;
        std::string rnnType;
        bool quantizedHiddenState = false;

        std::tie(rnnType, inputShapes, quantizedHiddenState) = obj.param;

        auto batchSize  = inputShapes[0];
        auto inputSize  = inputShapes[1];
        auto hiddenSize = inputShapes[2];

        std::ostringstream result;

        result << "Type=" << rnnType << "_";

        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << ov::test::utils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1 ? "_" : "");
            }
            result << "}_";
        }

        result << "quantizedHiddenState=" << quantizedHiddenState;

        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        const auto& funcInputs = function->inputs();
        const auto& shapeX = targetInputStaticShapes[0];
        const auto& shapeH = targetInputStaticShapes[1];

        ov::Tensor tensorX = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), shapeX, 1, 0, 16);
        ov::Tensor tensorH = utils::create_and_fill_tensor(funcInputs[1].get_element_type(), shapeH, 1, 0, 16);

        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorX});
        inputs.insert({funcInputs[1].get_node_shared_ptr(), tensorH});

        if (hasCell) {
            const auto& shapeC = targetInputStaticShapes[cellIdx];
            ov::Tensor tensorC = utils::create_and_fill_tensor(funcInputs[cellIdx].get_element_type(), shapeC, 2, -1, 128, 2);
            inputs.insert({funcInputs[cellIdx].get_node_shared_ptr(), tensorC});
        }
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        selectedType = "ref_any_I8";

        std::vector<InputShape> inputShapes;
        std::string rnnType;
        bool quantizedHiddenState = false;

        std::tie(rnnType, inputShapes, quantizedHiddenState) = this->GetParam();

        if (rnnType != "LSTMSequence") // remove cell input for non-cell rnn types
            inputShapes.erase(inputShapes.begin() + cellIdx);

        init_input_shapes(inputShapes);

        const auto inputSize  = targetStaticShapes.front()[0][2];
        const auto hiddenSize = targetStaticShapes.front()[1][2];
        const size_t numDirections = 1;
        const size_t numOfGates     = rnnType == "LSTMSequence"   ? 4 : 3;
        const size_t numOfBiasGates = rnnType == "LBRGRUSequence" ? numOfGates + 1 : numOfGates;

        const auto ngPrec = element::f32;
        std::shared_ptr<Node> H;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrec, shape));

        auto makeDataFQ = [](const ov::Output<Node>& input) {
            const auto fqLevels = 256;
            return ngraph::builder::makeFakeQuantize(input, ov::element::f32, fqLevels, {},
                                                      {-128.f/127}, {1.f},
                                                      {-128.f/127}, {1.f});
        };

        auto X_FQ = makeDataFQ(inputParams[0]);

        if (quantizedHiddenState) {
            H = makeDataFQ(inputParams[1]);
        } else {
            H = ngraph::builder::makeConstant(ov::element::f32, inputDynamicShapes[1].get_shape(),  {}, true, 1.f, -1.f);
        }

        auto W = ngraph::builder::makeConstant(ov::element::f32, {numDirections, numOfGates     * hiddenSize, inputSize},  {}, true, 1.f, -1.f);
        auto R = ngraph::builder::makeConstant(ov::element::f32, {numDirections, numOfGates     * hiddenSize, hiddenSize}, {}, true, 1.f, -1.f);
        auto B = ngraph::builder::makeConstant(ov::element::f32, {numDirections, numOfBiasGates * hiddenSize},             {}, true, 0.1f, -0.1f);

        auto makeWeightsFQ = [](const std::shared_ptr<Node> weight) {
            const auto fqLevelsW = 255;
            return ngraph::builder::makeFakeQuantize(weight, ov::element::f32,
                                                     fqLevelsW, std::vector<size_t>{},
                                                     {-127.f/63}, {127.f/63},
                                                     {-127.f/63}, {127.f/63});
        };

        auto W_FQ = makeWeightsFQ(W);
        auto R_FQ = makeWeightsFQ(R);

        std::shared_ptr<ov::Node> rnnCellOp;

        // fill sequence_length constant with max sequence length values
        const auto batchSize  = targetStaticShapes.front()[0][0];
        const auto maxSeqLen  = targetStaticShapes.front()[0][1];
        std::vector<int> lengths(batchSize, static_cast<int>(maxSeqLen));
        auto seq_lengths = ov::opset1::Constant::create(element::i64, Shape{batchSize}, lengths);

        if (rnnType == "LSTMSequence") {
            hasCell = true;
            auto C = inputParams[cellIdx];
            rnnCellOp = std::make_shared<ov::op::v5::LSTMSequence>(
                X_FQ, H, C, seq_lengths, W_FQ, R_FQ, B,
                hiddenSize, op::RecurrentSequenceDirection::FORWARD);
        } else if (rnnType == "GRUSequence") {
            rnnCellOp = std::make_shared<ov::op::v5::GRUSequence>(
                X_FQ, H, seq_lengths, W_FQ, R_FQ, B,
                hiddenSize, op::RecurrentSequenceDirection::FORWARD);
        } else if (rnnType == "LBRGRUSequence") {
            const std::vector<std::string> activations{"sigmoid", "tanh"};
            const std::vector<float> activations_alpha, activations_beta;
            rnnCellOp = std::make_shared<ov::op::v5::GRUSequence>(
                X_FQ, H, seq_lengths, W_FQ, R_FQ, B,
                hiddenSize, op::RecurrentSequenceDirection::FORWARD,
                activations, activations_alpha, activations_beta, 0.f, true);
        } else {
            OPENVINO_THROW("Unexpected offset type");
        }

        if (maxSeqLen > 1)
            abs_threshold = 0.05; // RNN int8 computation is expected to affect the accuracy, especially when sequence_length > 1

        function = makeNgraphFunction(ngPrec, inputParams, rnnCellOp, "ConvertFqRnnToQuantizedRnn");
    }
private:
    static const size_t cellIdx = 2;
    bool hasCell = false;
};

TEST_P(ConvertFqRnnToQuantizedRnn, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RNNSeq");
}

namespace {

const std::vector<std::vector<InputShape>> staticShapesLSTM = {
    {   // seq len > 1
        { {}, { {2, 5, 10} } },  // X
        { {}, { {2, 1, 4}} },    // H
        { {}, { {2, 1, 4}} },    // C
    },
    {   // seq len = 1
        { {}, { {2, 1, 5} } },   // X
        { {}, { {2, 1, 1}} },    // H
        { {}, { {2, 1, 1}} },    // C
    },
};

std::vector<bool> quantizedHiddenStateParam{true, false};

INSTANTIATE_TEST_SUITE_P(smoke_static, ConvertFqRnnToQuantizedRnn,
                         ::testing::Combine(::testing::Values("LSTMSequence", "GRUSequence"),
                                            // "LBRGRUSequence", // enable after implemented in oneDNN
                                            ::testing::ValuesIn(staticShapesLSTM),
                                            ::testing::ValuesIn(quantizedHiddenStateParam)),
                         ConvertFqRnnToQuantizedRnn::getTestCaseName);
} // namespace

}  // namespace test
}  // namespace ov
