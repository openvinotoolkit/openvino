// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/lstm_sequence.hpp"
#include <transformations/bidirectional_sequences_decomposition.hpp>

namespace LayerTestsDefinitions {

    std::string LSTMSequenceTest::getTestCaseName(const testing::TestParamInfo<LSTMSequenceParams> &obj) {
        //bool should_decompose;
        size_t seq_lenghts;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(seq_lenghts, batch, hidden_size, input_size, activations, clip, direction, netPrecision,
                 targetDevice) = obj.param;
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                        {4 * hidden_size, hidden_size}, {4 * hidden_size}},
        };
        std::ostringstream result;
        result << "seq_lenghts" << seq_lenghts << "_";
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "activations=" << CommonTestUtils::vec2str(activations) << "_";
        result << "direction=" << direction << "_";
        result << "clip=" << clip << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        return result.str();
    }

    void LSTMSequenceTest::SetUp() {
        size_t seq_lenghts;
        // bool should_decompose;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::tie(seq_lenghts, batch, hidden_size, input_size, activations, clip, direction, netPrecision,
                 targetDevice) = this->GetParam();
        size_t num_directions = direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, seq_lenghts, input_size}, {batch, num_directions, hidden_size}, {batch, num_directions, hidden_size},
                 {batch}, {num_directions, 4 * hidden_size, input_size}, {num_directions, 4 * hidden_size, hidden_size}, {num_directions, 4 * hidden_size}},
        };
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1], inputShapes[2]});
        std::vector<ngraph::Shape> WRB = {inputShapes[4], inputShapes[5], inputShapes[6], inputShapes[3]};
        auto lstm_sequence = ngraph::builder::makeLSTM(ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)),
                                                       WRB, hidden_size, activations, {}, {}, clip, true, direction);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lstm_sequence->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(lstm_sequence->output(1)),
                                     std::make_shared<ngraph::opset1::Result>(lstm_sequence->output(2))};
        function = std::make_shared<ngraph::Function>(results, params, "lstm_sequence");
    }


    TEST_P(LSTMSequenceTest, CompareWithRefs) {
        Run();
    };
}  // namespace LayerTestsDefinitions
