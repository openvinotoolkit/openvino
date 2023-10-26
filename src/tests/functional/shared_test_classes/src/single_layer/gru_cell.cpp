// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include "shared_test_classes/single_layer/gru_cell.hpp"

namespace LayerTestsDefinitions {

using ngraph::helpers::InputLayerType;

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
    InputLayerType WType;
    InputLayerType RType;
    InputLayerType BType;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip,
            linear_before_reset, WType, RType, BType, netPrecision, targetDevice) = obj.param;
    inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                    {3 * hidden_size, hidden_size}, {(linear_before_reset? 4 : 3) * hidden_size}},
    };
    std::ostringstream result;
    result << "decomposition" << should_decompose << "_";
    result << "batch=" << batch << "_";
    result << "hidden_size=" << hidden_size << "_";
    result << "input_size=" << input_size << "_";
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "activations=" << ov::test::utils::vec2str(activations) << "_";
    result << "clip=" << clip << "_";
    result << "linear_before_reset=" << linear_before_reset << "_";
    result << "WType=" << WType << "_";
    result << "RType=" << RType << "_";
    result << "BType=" << BType << "_";
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
    InputLayerType WType;
    InputLayerType RType;
    InputLayerType BType;
    InferenceEngine::Precision netPrecision;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, linear_before_reset,
            WType, RType, BType, netPrecision, targetDevice) = this->GetParam();

    std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                    {3 * hidden_size, hidden_size}, {(linear_before_reset? 4 : 3) * hidden_size}},
    };
    std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};

    std::shared_ptr<ov::Node> W;
    if (WType == InputLayerType::PARAMETER) {
        const auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, WRB[0]);
        W = param;
        params.push_back(param);
    } else {
        W = ngraph::builder::makeConstant<float>(ngPrc, WRB[0], {}, true);
    }

    std::shared_ptr<ov::Node> R;
    if (RType == InputLayerType::PARAMETER) {
        const auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, WRB[1]);
        R = param;
        params.push_back(param);
    } else {
        R = ngraph::builder::makeConstant<float>(ngPrc, WRB[1], {}, true);
    }

    std::shared_ptr<ov::Node> B;
    if (BType == InputLayerType::PARAMETER) {
        const auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, WRB[2]);
        B = param;
        params.push_back(param);
    } else {
        B = ngraph::builder::makeConstant<float>(ngPrc, WRB[2], {}, true);
    }

    auto gru_cell = std::make_shared<ov::op::v3::GRUCell>(params[0], params[1], W, R, B, hidden_size, activations,
                                                          activations_alpha, activations_beta, clip,
                                                          linear_before_reset);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0))};
    function = std::make_shared<ngraph::Function>(results, params, "gru_cell");
    if (should_decompose) {
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::GRUCellDecomposition>();
        m.run_passes(function);
    }
}
}  // namespace LayerTestsDefinitions
