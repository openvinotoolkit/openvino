// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "shared_test_classes/single_layer/lstm_cell_basic.hpp"

namespace LayerTestsDefinitions {

std::string LSTMCellBasicTest::getTestCaseName(const testing::TestParamInfo<LSTMCellBasicParams> &obj) {
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> additionalConfig;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, netPrecision,
            targetDevice, additionalConfig) = obj.param;
    std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                    {4 * hidden_size, hidden_size}, {4 * hidden_size}},
    };
    std::ostringstream result;
    result << "decomposition" << should_decompose << "_";
    result << "batch=" << batch << "_";
    result << "hidden_size=" << hidden_size << "_";
    result << "input_size=" << input_size << "_";
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "activations=" << ov::test::utils::vec2str(activations) << "_";
    result << "clip=" << clip << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    for (const auto& configEntry : additionalConfig) {
        result << configEntry.first << ", " << configEntry.second << ";";
    }
    return result.str();
}

void LSTMCellBasicTest::SetUp() {
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additionalConfig;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, netPrecision,
            targetDevice, additionalConfig) = this->GetParam();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                    {4 * hidden_size, hidden_size}, {4 * hidden_size}},
    };
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1])),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[2]))};
    ov::OutputVector paramsOuts;
    for (auto&& param : params)
        paramsOuts.push_back(param);

    std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5]};
    auto lstm_cell = ngraph::builder::makeLSTM(paramsOuts, WRB, hidden_size, activations, {}, {}, clip);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lstm_cell->output(0)),
                                 std::make_shared<ngraph::opset1::Result>(lstm_cell->output(1))};
    function = std::make_shared<ngraph::Function>(results, params, "lstm_cell");
    if (should_decompose) {
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::LSTMCellDecomposition>();
        m.run_passes(function);
    }
}
}  // namespace LayerTestsDefinitions
