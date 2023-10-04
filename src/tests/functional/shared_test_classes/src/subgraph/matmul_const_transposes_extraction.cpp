// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_const_transposes_extraction.hpp"
#include "shared_test_classes/subgraph/matmul_const_transposes_extraction.hpp"
#include "ov_models/builders.hpp"
#include <exec_graph_info.hpp>

namespace SubgraphTestsDefinitions {

using namespace ngraph;

std::string MatMulConstTransposesExtractionTest::getTestCaseName(const testing::TestParamInfo<MatMulConstTransposesExtractionTestParams> &obj) {
    MatMulConstTransposesExtractionTestShapeParams shape_params;
    std::string device;
    std::tie(shape_params, std::ignore, device) = obj.param;
    std::ostringstream results;

    results << "input=" << shape_params.input_shape << "_";
    results << "weights=" << shape_params.weights_shape << "_";
    results << "transB=" << std::boolalpha << shape_params.trans_b << "_";
    results << "dev=" << device;
    return results.str();
}

void MatMulConstTransposesExtractionTest::SetUp() {
    MatMulConstTransposesExtractionTestShapeParams shape_params;
    element::Type type = element::f32;
    bool can_be_fused;
    std::tie(shape_params, can_be_fused, targetDevice) = GetParam();

    const auto& input_shape = shape_params.input_shape;
    const auto& weights_shape = shape_params.weights_shape;

    auto param = std::make_shared<opset8::Parameter>(type, input_shape);
    auto weights = opset8::Constant::create(type, weights_shape, {0.5});
    auto matmul = std::make_shared<opset8::MatMul>(param, weights, false, shape_params.trans_b);
    function = std::make_shared<Function>(matmul, ParameterVector{param});

    auto transformed_function = clone_function(*function);
    pass::Manager manager;
    manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
    manager.run_passes(transformed_function);

    bool functions_equal;
    auto orig_function = clone_function(*function);
    std::tie(functions_equal, std::ignore) = compare_functions(transformed_function, orig_function, true);
    if (can_be_fused) {
        ASSERT_FALSE(functions_equal);
    } else {
        ASSERT_TRUE(functions_equal);
    }
}

std::string QuantizedMatMulConstTransposesExtractionTest::getTestCaseName(
        const testing::TestParamInfo<MatMulConstTransposesExtractionTestParams> &obj) {
    MatMulConstTransposesExtractionTestShapeParams params;
    std::string device;
    std::tie(params, std::ignore, device) = obj.param;
    std::ostringstream results;

    results << "input=" << params.input_shape << "_"
        "weights=" << params.weights_shape << "_"
        "dev=" << device;
    return results.str();
}

void QuantizedMatMulConstTransposesExtractionTest::SetUp() {
    MatMulConstTransposesExtractionTestShapeParams params;
    bool can_be_fused;
    std::tie(params, can_be_fused, targetDevice) = GetParam();

    const auto& input_shape = params.input_shape;
    auto weights_shape = params.weights_shape;

    element::Type type = element::f32;
    auto param = std::make_shared<opset8::Parameter>(type, input_shape);
    std::shared_ptr<Node> input;
    std::shared_ptr<Node> weights = opset8::Constant::create(type, weights_shape, {0.5});
    auto low = opset8::Constant::create(type, {1}, {-2});
    auto high = opset8::Constant::create(type, {1}, {2});
    input = std::make_shared<opset8::FakeQuantize>(param, low, high, low, high, 256);
    weights = std::make_shared<opset8::FakeQuantize>(weights, low, high, low, high, 255);
    auto matmul = std::make_shared<opset8::MatMul>(input, weights, false, false);
    function = std::make_shared<Function>(matmul, ParameterVector{param});

    auto transformed_function = clone_function(*function);
    pass::Manager manager;
    manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
    manager.run_passes(transformed_function);

    bool functions_equal;
    auto orig_function = clone_function(*function);
    std::tie(functions_equal, std::ignore) = compare_functions(transformed_function, orig_function, true);
    if (can_be_fused) {
        ASSERT_FALSE(functions_equal);
    } else {
        ASSERT_TRUE(functions_equal);
    }
}

void QuantizedMatMulConstTransposesExtractionTest::TearDown() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
    int ops_found = 0;
    for (const auto& node : runtime_function->get_ordered_ops()) {
        const auto& layer_type = node->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
        if (layer_type == "FullyConnected" || layer_type == "MatMul") {
            ops_found++;
            auto inputs = node->input_values();
            ASSERT_EQ(element::u8, inputs[0].get_element_type());
            ASSERT_EQ(element::i8, inputs[1].get_element_type());
        }
    }
    ASSERT_GT(ops_found, 0);
}
} // namespace SubgraphTestsDefinitions
