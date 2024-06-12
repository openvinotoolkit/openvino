// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_split_decompose.hpp"

#include "common_test_utils/graph_comparator.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "transformations/common_optimizations/matmul_split_decomposition.hpp"
#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {

std::string MatMulSplitDecompose::getTestCaseName(const testing::TestParamInfo<MatMulSplitDecomposeParams>& obj) {
    MatMulSplitDecomposeShapeParams shape_params;
    std::string device;
    std::tie(shape_params, device) = obj.param;
    std::ostringstream results;

    results << "input=" << shape_params.input_shape << "_";
    results << "weights=" << shape_params.weights_shape << "_";
    results << "transB=" << std::boolalpha << shape_params.trans_b << "_";
    results << "bias=" << shape_params.bias_shape << "_";
    results << "reshape=" << shape_params.reshape_shape << "_";
    results << "dev=" << device;
    return results.str();
}

void MatMulSplitDecompose::SetUp() {
    MatMulSplitDecomposeShapeParams shape_params;
    element::Type precision = element::f32;
    std::tie(shape_params, targetDevice) = GetParam();

    const auto& input_shape = shape_params.input_shape;
    const auto& weights_shape = shape_params.weights_shape;
    const auto& bias_shape = shape_params.bias_shape;
    const auto& reshape_shape = shape_params.reshape_shape;

    auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);

    std::vector<float> weights_vals(shape_size(weights_shape), 2.0f);
    weights_vals = ov::test::utils::generate_float_numbers(shape_size(weights_shape), -0.1f, 0.1f);
    // std::iota (std::begin(weights_vals), std::end(weights_vals), 0.f);
    // std::copy(weights_vals.begin(), weights_vals.end(), std::ostream_iterator<float>(std::cout, ", "));
    // std::cout << "\n\n";
    auto weights = ov::op::v0::Constant::create(precision, weights_shape, weights_vals);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, shape_params.trans_b);

    std::vector<float> bias_vals = {0.0};
    bias_vals = ov::test::utils::generate_float_numbers(shape_size(bias_shape), -0.1f, 0.1f);
    // std::copy(bias_vals.begin(), bias_vals.end(), std::ostream_iterator<float>(std::cout, ", "));
    // std::cout << "\n\n";
    auto bias = ov::op::v0::Constant::create(precision, bias_shape, bias_vals);
    auto add = std::make_shared<ov::op::v1::Add>(matmul, bias);

    auto reshape_const = ov::op::v0::Constant::create(element::i64, {reshape_shape.size()}, reshape_shape.data());
    auto reshape = std::make_shared<ov::op::v1::Reshape>(add, reshape_const, false);

    auto transpose_const = ov::op::v0::Constant::create(element::i64, {5}, {2, 0, 3, 1, 4});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);

    auto const_zero = ov::op::v0::Constant::create(element::i64, {}, {0});
    auto const_one = ov::op::v0::Constant::create(element::i64, {}, {1});
    auto const_two = ov::op::v0::Constant::create(element::i64, {}, {2});
    auto gather_0 = std::make_shared<ov::op::v1::Gather>(transpose, const_zero/*indices*/, const_zero/*axis*/);
    auto gather_1 = std::make_shared<ov::op::v1::Gather>(transpose, const_one/*indices*/, const_zero/*axis*/);
    auto gather_2 = std::make_shared<ov::op::v1::Gather>(transpose, const_two/*indices*/, const_zero/*axis*/);

    auto mul_const = ov::op::v0::Constant::create(precision, {1}, {2.0});
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(gather_1, mul_const);

    auto mm_qk = std::make_shared<ov::op::v0::MatMul>(gather_0, mul2, false, true);
    auto softmax = std::make_shared<ov::op::v1::Softmax>(mm_qk);
    auto mm_v = std::make_shared<ov::op::v0::MatMul>(softmax, gather_2, false, false);

    functionRefs = std::make_shared<Model>(OutputVector{mm_v}, ParameterVector{param});
    ov::pass::Serialize serializer("matmul_gathers_ref.xml", "matmul_gathers_ref.bin");
    serializer.run_on_model(functionRefs);

    function = functionRefs->clone();
    ov::pass::Manager manager;
    // apply
    manager.register_pass<ov::pass::MatmulSplitDecomposition>();
    manager.run_passes(function);
    ov::pass::Serialize serializer2("matmul_gathers.xml", "matmul_gathers.bin");
    serializer2.run_on_model(function);
}

}  // namespace test
}  // namespace ov
