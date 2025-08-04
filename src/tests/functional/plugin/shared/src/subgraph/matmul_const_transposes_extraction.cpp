// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_const_transposes_extraction.hpp"

#include "common_test_utils/graph_comparator.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/subgraph/matmul_const_transposes_extraction.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace test {

std::string MatMulConstTransposesExtractionTest::getTestCaseName(
    const testing::TestParamInfo<MatMulConstTransposesExtractionTestParams>& obj) {
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

    auto param = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    auto weights = ov::op::v0::Constant::create(type, weights_shape, {0.5});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, shape_params.trans_b);
    function = std::make_shared<Model>(matmul, ParameterVector{param});

    auto transformed_function = function->clone();
    pass::Manager manager;
    manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
    manager.run_passes(transformed_function);

    bool functions_equal;
    auto orig_function = function->clone();
    std::tie(functions_equal, std::ignore) = compare_functions(transformed_function, orig_function, true);
    if (can_be_fused) {
        ASSERT_FALSE(functions_equal);
    } else {
        ASSERT_TRUE(functions_equal);
    }
}

std::string QuantizedMatMulConstTransposesExtractionTest::getTestCaseName(
    const testing::TestParamInfo<MatMulConstTransposesExtractionTestParams>& obj) {
    MatMulConstTransposesExtractionTestShapeParams params;
    std::string device;
    std::tie(params, std::ignore, device) = obj.param;
    std::ostringstream results;

    results << "input=" << params.input_shape
            << "_"
               "weights="
            << params.weights_shape
            << "_"
               "dev="
            << device;
    return results.str();
}

void QuantizedMatMulConstTransposesExtractionTest::SetUp() {
    MatMulConstTransposesExtractionTestShapeParams params;
    bool can_be_fused;
    std::tie(params, can_be_fused, targetDevice) = GetParam();

    const auto& input_shape = params.input_shape;
    auto weights_shape = params.weights_shape;

    element::Type type = element::f32;
    auto param = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    std::shared_ptr<Node> input;
    std::shared_ptr<Node> weights = ov::op::v0::Constant::create(type, weights_shape, {0.5});
    auto low = ov::op::v0::Constant::create(type, {1}, {-2});
    auto high = ov::op::v0::Constant::create(type, {1}, {2});
    input = std::make_shared<ov::op::v0::FakeQuantize>(param, low, high, low, high, 256);
    weights = std::make_shared<ov::op::v0::FakeQuantize>(weights, low, high, low, high, 255);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    function = std::make_shared<Model>(matmul, ParameterVector{param});

    auto transformed_function = function->clone();
    pass::Manager manager;
    manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
    manager.run_passes(transformed_function);

    bool functions_equal;
    auto orig_function = function->clone();
    std::tie(functions_equal, std::ignore) = compare_functions(transformed_function, orig_function, true);
    if (can_be_fused) {
        ASSERT_FALSE(functions_equal);
    } else {
        ASSERT_TRUE(functions_equal);
    }

    if (type == element::f32) {
        abs_threshold = 2e-7;
    }
}

void QuantizedMatMulConstTransposesExtractionTest::TearDown() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto runtime_function = compiledModel.get_runtime_model();
    int ops_found = 0;
    for (const auto& node : runtime_function->get_ordered_ops()) {
        const auto& layer_type = node->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
        if (layer_type == "FullyConnected" || layer_type == "MatMul") {
            ops_found++;
            auto inputs = node->input_values();
            ASSERT_EQ(element::u8, inputs[0].get_element_type());
            ASSERT_EQ(element::i8, inputs[1].get_element_type());
        }
    }
    ASSERT_GT(ops_found, 0);
}

}  // namespace test
}  // namespace ov
