// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include "shared_test_classes/single_layer/gru_cell.hpp"

namespace LayerTestsDefinitions {

std::string GRUCellTest::getTestCaseName(const testing::TestParamInfo<GRUCellParams> &obj) {
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    bool linear_before_reset;
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip,
            linear_before_reset, netPrecision, targetDevice) = obj.param;
    inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                    {3 * hidden_size, hidden_size}, {(linear_before_reset? 4 : 3) * hidden_size}},
    };
    std::ostringstream result;
    result << "decomposition" << should_decompose << "_";
    result << "batch=" << batch << "_";
    result << "hidden_size=" << hidden_size << "_";
    result << "input_size=" << input_size << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "activations=" << CommonTestUtils::vec2str(activations) << "_";
    result << "clip=" << clip << "_";
    result << "linear_before_reset=" << linear_before_reset << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    return result.str();
}

void GRUCellTest::SetUp() {
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    bool linear_before_reset;
    InferenceEngine::Precision netPrecision;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, linear_before_reset,
            netPrecision, targetDevice) = this->GetParam();

    std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                    {3 * hidden_size, hidden_size}, {(linear_before_reset? 4 : 3) * hidden_size}},
    };

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
    std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
    auto gru_cell = ngraph::builder::makeGRU(
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)),
            WRB, hidden_size, activations, {}, {}, clip, linear_before_reset);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0))};
    function = std::make_shared<ngraph::Function>(results, params, "gru_cell");
    if (should_decompose) {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::GRUCellDecomposition>();
        m.run_passes(function);
    }
}
}  // namespace LayerTestsDefinitions
