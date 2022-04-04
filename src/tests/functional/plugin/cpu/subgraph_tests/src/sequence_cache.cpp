// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace SubgraphTestsDefinitions {

using RnnSeqParams = std::tuple<std::vector<std::string>, // activations 1
                                std::vector<std::string>, // activations 2
                                size_t,                   // batch_size
                                size_t>;                  // seq_length

class SequenceCacheCPUTest : public testing::WithParamInterface<RnnSeqParams>, public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RnnSeqParams> &obj) {
        std::vector<std::string> activations1, activations2;
        size_t batch_size, seq_length;
        std::tie(activations1, activations2, batch_size, seq_length) = obj.param;

        std::ostringstream result;
        result << "activations_1=" << CommonTestUtils::vec2str(activations1)  << "_";
        result << "activations_2=" << CommonTestUtils::vec2str(activations2)  << "_";
        result << "batch_size=" << batch_size << ", seq_length=" << seq_length;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        std::vector<std::string> activations1, activations2;
        size_t batch_size, seq_length;
        std::tie(activations1, activations2, batch_size, seq_length) = this->GetParam();

        const size_t numDirections = 1;
        const size_t hidden_size = 10, input_size = 10;
        const float clip = 0.0f;
        const ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD;

        const ElementType netPrecision = ElementType::f32;

        // dynamic shapes
        ov::PartialShape X_shape({-1, -1, -1});
        ov::PartialShape H_shape({-1, -1, -1});
        ov::PartialShape seq_len_shape({-1});
        inputDynamicShapes.push_back(X_shape);
        inputDynamicShapes.push_back(H_shape);
        inputDynamicShapes.push_back(seq_len_shape);

        std::vector<ov::Shape> weightShape;
        ov::Shape W_shape(std::vector<size_t>{numDirections, hidden_size, input_size});
        ov::Shape R_shape(std::vector<size_t>{numDirections, hidden_size, hidden_size});
        ov::Shape B_shape = std::vector<size_t>{numDirections, hidden_size};
        weightShape.push_back(W_shape);
        weightShape.push_back(R_shape);
        weightShape.push_back(B_shape);

        // funciton creation
        IE_ASSERT(inputDynamicShapes.size() == paramsNum);

        std::vector<ov::element::Type> types(inputDynamicShapes.size(), netPrecision);
        types.back() = ElementType::i64;
        types.resize(types.size() * 2);
        for (size_t i = paramsNum; i < types.size(); i++) {
            types[i] = types[i - paramsNum];
        }

        inputDynamicShapes.resize(inputDynamicShapes.size() * 2);
        for (size_t i = paramsNum; i < inputDynamicShapes.size(); i++) {
            inputDynamicShapes[i] = inputDynamicShapes[i - paramsNum];
        }

        auto params = ngraph::builder::makeDynamicParams(types, inputDynamicShapes);

        for (size_t i = 0; i < params.size(); i++) {
            params[i]->set_friendly_name("input_" + std::to_string(i));
        }

        std::shared_ptr<ov::Node> rnn1 = ngraph::builder::makeRNN(ov::OutputVector(params.begin(), params.begin() + paramsNum),
                                                                  weightShape,
                                                                  hidden_size,
                                                                  activations1,
                                                                  {},
                                                                  {},
                                                                  clip,
                                                                  true,
                                                                  direction,
                                                                  ngraph::helpers::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM);
        std::shared_ptr<ov::Node> rnn2 = ngraph::builder::makeRNN(ov::OutputVector(params.begin() + paramsNum, params.end()),
                                                                  weightShape,
                                                                  hidden_size,
                                                                  activations2,
                                                                  {},
                                                                  {},
                                                                  clip,
                                                                  true,
                                                                  direction,
                                                                  ngraph::helpers::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM);

        ov::OutputVector results;
        for (size_t i = 0; i < rnn1->get_output_size(); i++) {
            results.push_back(rnn1->output(i));
        }
        for (size_t i = 0; i < rnn2->get_output_size(); i++) {
            results.push_back(rnn2->output(i));
        }

        function = std::make_shared<ov::Model>(results, params, "SequenceCacheCPUTest");

        std::vector<ov::Shape> targetShapes;
        targetShapes.push_back({batch_size, seq_length, input_size});
        targetShapes.push_back({batch_size, numDirections, hidden_size});
        targetShapes.push_back({batch_size});

        targetShapes.resize(targetShapes.size() * 2);
        for (size_t i = paramsNum; i < targetShapes.size(); i++) {
            targetShapes[i] = targetShapes[i - paramsNum];
        }

        targetStaticShapes.push_back(targetShapes);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);

        const size_t batchSize = targetInputStaticShapes[0][0];
        const int64_t maxSeqLen = targetInputStaticShapes[0][1];

        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < 2; i++) {
            const auto& seqLenInput = inputs.find(funcInputs[paramsNum * i + seqLengthInIdx].get_node_shared_ptr());
            if (seqLenInput == inputs.end())
                throw std::runtime_error("Could not find Sequence length input.");

            auto lenData = seqLenInput->second.data<ov::element_type_traits<ElementType::i64>::value_type>();
            std::fill(lenData, lenData + batchSize, maxSeqLen);
        }
    }

private:
    const size_t seqLengthInIdx = 2;
    const size_t paramsNum = 3;
};

TEST_P(SequenceCacheCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_SequenceCacheCPUTest_two_same_rnn, SequenceCacheCPUTest,
            ::testing::Combine(::testing::Values(std::vector<std::string>{"sigmoid"}),
                               ::testing::Values(std::vector<std::string>{"sigmoid"}),
                               ::testing::Values(10),
                               ::testing::Values(5)),
            SequenceCacheCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SequenceCacheCPUTest_same_rnn_diff_activ, SequenceCacheCPUTest,
            ::testing::Combine(::testing::Values(std::vector<std::string>{"sigmoid"}),
                               ::testing::Values(std::vector<std::string>{"tanh"}),
                               ::testing::Values(10),
                               ::testing::Values(5)),
            SequenceCacheCPUTest::getTestCaseName);

} // namespace SubgraphTestsDefinitions
