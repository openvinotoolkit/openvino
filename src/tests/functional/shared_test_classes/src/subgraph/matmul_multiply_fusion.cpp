// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_multiply_fusion.hpp"

#include "common_test_utils/graph_comparator.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "transformations/common_optimizations/matmul_multiply_fusion.hpp"

namespace ov {
namespace test {

std::string MatMulMultiplyFusion::getTestCaseName(const testing::TestParamInfo<MatMulMultiplyFusionParams>& obj) {
    MatMulMultiplyFusionShapeParams shape_params;
    std::string device;
    std::tie(shape_params, std::ignore, device) = obj.param;
    std::ostringstream results;

    results << "input=" << shape_params.input_shape << "_";
    results << "weights=" << shape_params.weights_shape << "_";
    results << "transB=" << std::boolalpha << shape_params.trans_b << "_";
    results << "const=" << shape_params.const_shape << "_";
    results << "dev=" << device;
    return results.str();
}

void MatMulMultiplyFusion::SetUp() {
    MatMulMultiplyFusionShapeParams shape_params;
    element::Type precision = element::f32;
    bool can_be_fused;
    std::tie(shape_params, can_be_fused, targetDevice) = GetParam();

    const auto& input_shape = shape_params.input_shape;
    const auto& weights_shape = shape_params.weights_shape;
    const auto& const_shape = shape_params.const_shape;

    auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
    auto weights = ov::op::v0::Constant::create(precision, weights_shape, {0.5});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, shape_params.trans_b);
    auto mul_const = ov::op::v0::Constant::create(precision, const_shape, {2.0});
    auto mul = std::make_shared<ov::op::v1::Multiply>(matmul, mul_const);
    function = std::make_shared<Model>(OutputVector{mul}, ParameterVector{param});

    auto transformed_function = function->clone();
    pass::Manager manager;
    manager.register_pass<ov::pass::MatMulMultiplyFusion>();
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

std::string QuantizedMatMulMultiplyFusion::getTestCaseName(
    const testing::TestParamInfo<MatMulMultiplyFusionParams>& obj) {
    MatMulMultiplyFusionShapeParams shape_params;
    std::string device;
    std::tie(shape_params, std::ignore, device) = obj.param;
    std::ostringstream results;

    results << "input=" << shape_params.input_shape << "_";
    results << "weights=" << shape_params.weights_shape << "_";
    results << "transB=" << std::boolalpha << shape_params.trans_b << "_";
    results << "const=" << shape_params.const_shape << "_";
    results << "dev=" << device;
    return results.str();
}

void QuantizedMatMulMultiplyFusion::SetUp() {
    MatMulMultiplyFusionShapeParams shape_params;
    element::Type precision = element::f32;
    bool can_be_fused;
    std::tie(shape_params, can_be_fused, targetDevice) = GetParam();

    const auto& input_shape = shape_params.input_shape;
    auto weights_shape = shape_params.weights_shape;
    const auto& const_shape = shape_params.const_shape;

    auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
    auto low = ov::op::v0::Constant::create(precision, {1}, {-2});
    auto high = ov::op::v0::Constant::create(precision, {1}, {2});
    auto input_fq = std::make_shared<ov::op::v0::FakeQuantize>(param, low, high, low, high, 256);
    std::shared_ptr<Node> weights = ov::op::v0::Constant::create(precision, weights_shape, {0.5});
    weights = std::make_shared<ov::op::v0::FakeQuantize>(weights, low, high, low, high, 255);
    if (shape_params.trans_b) {
        std::vector<int> perm(weights_shape.size(), 0);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(*(perm.end() - 2), *(perm.end() - 1));
        auto perm_const = ov::op::v0::Constant::create(element::i32, {perm.size()}, perm);
        weights = std::make_shared<ov::op::v1::Transpose>(weights, perm_const);
    }
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input_fq, weights);
    auto mul_const = ov::op::v0::Constant::create(precision, const_shape, {2});
    auto mul = std::make_shared<ov::op::v1::Multiply>(matmul, mul_const);
    function = std::make_shared<Model>(OutputVector{mul}, ParameterVector{param});

    auto transformed_function = function->clone();
    pass::Manager manager;
    manager.register_pass<ov::pass::MatMulMultiplyFusion>();
    manager.run_passes(transformed_function);

    bool functions_equal;
    auto orig_function = function->clone();
    std::tie(functions_equal, std::ignore) = compare_functions(transformed_function, orig_function, true);
    if (can_be_fused) {
        ASSERT_FALSE(functions_equal);
    } else {
        ASSERT_TRUE(functions_equal);
    }
    if (precision == element::f32) {
        abs_threshold = 4e-7;
    }
}

void QuantizedMatMulMultiplyFusion::TearDown() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto get_layer_type = [](const std::shared_ptr<ov::Node>& node) -> const std::string& {
        const auto& rt_info = node->get_rt_info();
        auto it = rt_info.find(ov::exec_model_info::LAYER_TYPE);
        OPENVINO_ASSERT(it != rt_info.end());
        return it->second.as<std::string>();
    };

    auto runtime_function = compiledModel.get_runtime_model();
    int ops_found = 0;
    for (const auto& node : runtime_function->get_ordered_ops()) {
        const auto& layer_type = get_layer_type(node);
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
