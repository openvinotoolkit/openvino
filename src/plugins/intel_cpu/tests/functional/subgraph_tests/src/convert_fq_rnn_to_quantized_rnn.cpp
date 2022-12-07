// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/output_vector.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/runtime/tensor.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;
using namespace ov;

namespace SubgraphTestsDefinitions {

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
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << CommonTestUtils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1 ? "_" : "");
            }
            result << "}_";
        }

        result << "quantizedHiddenState=" << quantizedHiddenState;

        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        const auto& funcInputs = function->inputs();
        const auto& shapeX = targetInputStaticShapes[0];
        const auto& shapeH = targetInputStaticShapes[1];

        // @todo update arguments after random data generation is fixed
        ov::Tensor tensorX = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), shapeX, 1*128, 0, 128);
        ov::Tensor tensorH = utils::create_and_fill_tensor(funcInputs[1].get_element_type(), shapeH, 1*128, 0, 128);

        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorX});
        inputs.insert({funcInputs[1].get_node_shared_ptr(), tensorH});

        if (hasCell) {
            const auto& shapeC = targetInputStaticShapes[2];
            ov::Tensor tensorC = utils::create_and_fill_tensor(funcInputs[2].get_element_type(), shapeC, 2*128, -1, 128, 2);
            inputs.insert({funcInputs[2].get_node_shared_ptr(), tensorC});
        }

        const size_t batchSize = targetInputStaticShapes[0][0];
        const int maxSeqLen = static_cast<int>(targetInputStaticShapes[0][1]);
        const auto& shapeSeqLen = targetInputStaticShapes[seqLenIdx];
        ov::Tensor tensorSeqLen{funcInputs[seqLenIdx].get_element_type(), shapeSeqLen};
        auto data = tensorSeqLen.data<ov::element_type_traits<ElementType::i64>::value_type>();
        std::fill(data, data + batchSize, maxSeqLen);
        inputs.insert({funcInputs[seqLenIdx].get_node_shared_ptr(), tensorSeqLen});
    }

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::string rnnType;
        bool quantizedHiddenState = false;

        std::tie(rnnType, inputShapes, quantizedHiddenState) = this->GetParam();

        if (rnnType == "GRUSequence")
            inputShapes.erase(inputShapes.begin() + 2);

        init_input_shapes(inputShapes);
        const auto inputSize  = targetStaticShapes.front()[0][2];
        const auto hiddenSize = targetStaticShapes.front()[1][2];
        const size_t numDirections = 1;
        const size_t numOfGates = rnnType == "LSTMSequence" ? 4 : 3;
        seqLenIdx = rnnType == "LSTMSequence" ? 3 : 2;

        const auto ngPrec = element::f32;
        ngraph::ParameterVector inputParams;
        std::shared_ptr<Node> H;

        inputParams = ngraph::builder::makeDynamicParams(ngPrec, inputDynamicShapes);
        inputParams.at(seqLenIdx)->set_element_type(ElementType::i64);

        const auto outputNodes = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(inputParams));


        auto makeDataFQ = [](const ngraph::Output<Node>& input) {
            const auto fqLevels = 256;
            return ngraph::builder::makeFakeQuantize(input, ngraph::element::f32, fqLevels, {},
                                                      {-128.f/127}, {1.f},
                                                      {-128.f/127}, {1.f});
        };

        auto X_FQ = makeDataFQ(outputNodes[0]);

        if (quantizedHiddenState) {
            H = makeDataFQ(outputNodes[1]);
        } else {
            H = ngraph::builder::makeConstant(ngraph::element::f32, inputDynamicShapes[1].get_shape(),  {}, true, 1.f, -1.f);
        }

        auto W = ngraph::builder::makeConstant(ngraph::element::f32, {numDirections, numOfGates * hiddenSize, inputSize},  {}, true, 1.f, -1.f);
        auto R = ngraph::builder::makeConstant(ngraph::element::f32, {numDirections, numOfGates * hiddenSize, hiddenSize}, {}, true, 1.f, -1.f);
        auto B = ngraph::builder::makeConstant(ngraph::element::f32, {numDirections, numOfGates * hiddenSize},             {}, true, 0.01f, -0.01f);

        auto makeWeightsFQ = [](const std::shared_ptr<Node> weight) {
            const auto fqLevelsW = 255;
            return ngraph::builder::makeFakeQuantize(weight, ngraph::element::f32,
                                                     fqLevelsW, std::vector<size_t>{},
                                                     {-127.f/63}, {127.f/63},
                                                     {-127.f/63}, {127.f/63});
        };

        auto W_FQ = makeWeightsFQ(W);
        auto R_FQ = makeWeightsFQ(R);

        std::shared_ptr<ov::Node> rnnCellOp;

        auto seq_lengths = outputNodes[seqLenIdx];

        if (rnnType == "LSTMSequence") {
            hasCell = true;
            auto C = outputNodes[2];
            rnnCellOp = std::make_shared<ov::op::v5::LSTMSequence>(
                X_FQ, H, C, seq_lengths, W_FQ, R_FQ, B,
                hiddenSize, op::RecurrentSequenceDirection::FORWARD);
        } else {
            rnnCellOp = std::make_shared<ov::op::v5::GRUSequence>(
                X_FQ, H, seq_lengths, W_FQ, R_FQ, B,
                hiddenSize, op::RecurrentSequenceDirection::FORWARD);
        }

        function = makeNgraphFunction(ngPrec, inputParams, rnnCellOp, "ConvertFqRnnToQuantizedRnn");
    }
private:
    bool hasCell = false;
    size_t seqLenIdx = 0;
};

TEST_P(ConvertFqRnnToQuantizedRnn, CompareWithRefs) {
    run();
}

namespace {

const std::vector<std::vector<InputShape>> staticShapesLSTM = {
    {   // seq len > 1
        { {}, { {2, 5, 10} } },  // X
        { {}, { {2, 1, 4}} },    // H
        { {}, { {2, 1, 4}} },    // C
        { {}, { {2} } }          // seq_length
    },
    {   // seq len = 1
        { {}, { {2, 1, 5} } },   // X
        { {}, { {2, 1, 1}} },    // H
        { {}, { {2, 1, 1}} },    // C
        { {}, { {2} } }          // seq_length
    },
};

std::vector<bool> quantizedHiddenStateParam{true, false};

INSTANTIATE_TEST_SUITE_P(smoke_static, ConvertFqRnnToQuantizedRnn,
                         ::testing::Combine(::testing::Values("LSTMSequence", "GRUSequence"),
                                            ::testing::ValuesIn(staticShapesLSTM),
                                            ::testing::ValuesIn(quantizedHiddenStateParam)),
                         ConvertFqRnnToQuantizedRnn::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicShapesLSTM = {
    {
        { {-1, 1, 10},                   // X
          { {1, 1, 10}, {16, 1, 10} } }, // Target shapes
        { {-1, 1, 4},                    // H
          { {1, 1, 4}, {16, 1, 4}} },    // Target shapes
        { {-1, 1, 4},                    // C
          { {1, 1, 4}, {16, 1, 4}} },    // Target shapes
        { {-1},                          // seq_length
          { {1}, {16} } }                // Target shapes
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, ConvertFqRnnToQuantizedRnn,
                         ::testing::Combine(::testing::Values("LSTMSequence", "GRUSequence"),
                                            ::testing::ValuesIn(dynamicShapesLSTM),
                                            ::testing::Values(true)),
                         ConvertFqRnnToQuantizedRnn::getTestCaseName);
} // namespace

} // namespace SubgraphTestsDefinitions
