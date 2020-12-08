// Copyright (C) 2020 Intel Corporation
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

#include <transformations/op_conversions/rnn_cell_decomposition.hpp>
#include "single_layer_tests/rnn_cell.hpp"

namespace LayerTestsDefinitions {

std::string RNNCellTest::getTestCaseName(const testing::TestParamInfo<RNNCellParams> &obj) {
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    float clip;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip,
             netPrecision, targetDevice) = obj.param;
    std::vector<std::vector<size_t>> inputShapes = {{batch, input_size}, {batch, hidden_size},
                                     {hidden_size, input_size}, {hidden_size, hidden_size}, {hidden_size}};
    std::ostringstream result;
    result << "decomposition" << should_decompose << "_";
    result << "batch=" << batch << "_";
    result << "hidden_size=" << hidden_size << "_";
    result << "input_size=" << input_size << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "activations=" << CommonTestUtils::vec2str(activations) << "_";
    result << "clip=" << clip << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    return result.str();
}

void RNNCellTest::SetUp() {
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    InferenceEngine::Precision netPrecision;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip,
            netPrecision, targetDevice) = this->GetParam();
    std::vector<std::vector<size_t>> inputShapes = {{batch, input_size}, {batch, hidden_size},
                                                    {hidden_size, input_size}, {hidden_size, hidden_size}, {hidden_size}};
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
    std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
    auto rnn_cell = ngraph::builder::makeRNN(
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)),
            WRB, hidden_size, activations, {}, {}, clip);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(rnn_cell)};
    function = std::make_shared<ngraph::Function>(results, params, "rnn_cell");
    if (should_decompose) {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::RNNCellDecomposition>();
        m.run_passes(function);
    }
}


TEST_P(RNNCellTest, CompareWithRefs) {
    Run();
};
}  // namespace LayerTestsDefinitions
