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

#include "single_layer_tests/lstm_cell.hpp"

namespace LayerTestsDefinitions {

std::string LSTMCellTest::getTestCaseName(const testing::TestParamInfo<LSTMCellParams> &obj) {
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(batch, hidden_size, input_size, activations, activations_alpha, activations_beta, clip,
             netPrecision, targetDevice) = obj.param;
    std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                    {4 * hidden_size, hidden_size}, {4 * hidden_size}},
    };
    std::ostringstream result;
    result << "batch=" << batch << "_";
    result << "hidden_size=" << hidden_size << "_";
    result << "input_size=" << input_size << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "activations=" << CommonTestUtils::vec2str(activations) << "_";
    result << "activations_alfa=" << CommonTestUtils::vec2str(activations_alpha) << "_";
    result << "activations_beta=" << CommonTestUtils::vec2str(activations_beta) << "_";
    result << "clip=" << clip << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    return result.str();
}

void LSTMCellTest::SetUp() {
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    InferenceEngine::Precision netPrecision;
    std::tie(batch, hidden_size, input_size, activations, activations_alpha, activations_beta, clip,
            netPrecision, targetDevice) = this->GetParam();
    std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                    {4 * hidden_size, hidden_size}, {4 * hidden_size}},
    };
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1], inputShapes[2]});
    auto W = ngraph::builder::makeConstant(ngPrc, inputShapes[3], {}, true);
    auto R = ngraph::builder::makeConstant(ngPrc, inputShapes[4], {}, true);
    auto B = ngraph::builder::makeConstant(ngPrc, inputShapes[5], {}, true);
    auto lstm_cell = std::make_shared<ngraph::opset4::LSTMCell>(params[0], params[1], params[2],
            W, R, B, hidden_size, activations, activations_alpha, activations_beta, clip);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lstm_cell->output(0)),
                                 std::make_shared<ngraph::opset1::Result>(lstm_cell->output(1))};
    function = std::make_shared<ngraph::Function>(results, params, "lstm_cell");
}


TEST_P(LSTMCellTest, CompareWithRefs) {
    Run();
};
}  // namespace LayerTestsDefinitions