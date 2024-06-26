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

inline void CheckNumberOfNodesWithType(std::shared_ptr<const ov::Model> function,
                                       const std::unordered_set<std::string>& nodeTypes,
                                       size_t expectedCount) {
    ASSERT_NE(nullptr, function);
    int num_ops = 0;
    for (const auto& node : function->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type = rt_info.find("layerType")->second.as<std::string>();
        if (nodeTypes.count(layer_type)) {
            num_ops++;
        }
    }
    ASSERT_EQ(num_ops, expectedCount);
}

std::string MatMulGatherDecompose::getTestCaseName(const testing::TestParamInfo<MatMulGatherDecomposeParams>& obj) {
    MatMulGatherDecomposeShapeParams shape_params;
    std::string device;
    std::tie(shape_params, device) = obj.param;
    std::ostringstream results;

    results << "input=" << shape_params.input_shape << "_";
    results << "weights=" << shape_params.weights_shape << "_";
    results << "trans_b=" << std::boolalpha << shape_params.trans_b << "_";
    results << "have_bias=" << std::boolalpha << shape_params.have_bias << "_";
    if (shape_params.have_bias) {
        results << "bias=" << shape_params.bias_shape << "_";
    }
    results << "reshape=" << shape_params.reshape_shape << "_";
    results << "dev=" << device;
    return results.str();
}

void MatMulGatherDecompose::SetUp() {
    MatMulGatherDecomposeShapeParams shape_params;
    element::Type precision = element::f32;
    std::tie(shape_params, targetDevice) = GetParam();

    const auto& input_shape = shape_params.input_shape;
    const auto& weights_shape = shape_params.weights_shape;
    const auto& have_bias = shape_params.have_bias;
    const auto& bias_shape = shape_params.bias_shape;
    const auto& reshape_shape = shape_params.reshape_shape;

    auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);

    std::vector<float> weights_vals(shape_size(weights_shape), 2.0f);
    weights_vals = ov::test::utils::generate_float_numbers(shape_size(weights_shape), -0.1f, 0.1f);
    auto weights = ov::op::v0::Constant::create(precision, weights_shape, weights_vals);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, shape_params.trans_b);

    auto reshape_productor = std::shared_ptr<ov::Node>(matmul);
    if (have_bias) {
        std::vector<float> bias_vals = {0.0};
        bias_vals = ov::test::utils::generate_float_numbers(shape_size(bias_shape), -0.1f, 0.1f);
        auto bias = ov::op::v0::Constant::create(precision, bias_shape, bias_vals);
        auto add = std::make_shared<ov::op::v1::Add>(matmul, bias);
        reshape_productor = std::shared_ptr<ov::Node>(add);
    }

    auto reshape_const = ov::op::v0::Constant::create(element::i64, {reshape_shape.size()}, reshape_shape.data());
    auto reshape = std::make_shared<ov::op::v1::Reshape>(reshape_productor, reshape_const, false);

    auto transpose_const = ov::op::v0::Constant::create(element::i64, {5}, {2, 0, 3, 1, 4});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);

    auto const_zero = ov::op::v0::Constant::create(element::i64, {}, {0});
    auto const_one = ov::op::v0::Constant::create(element::i64, {}, {1});
    auto const_two = ov::op::v0::Constant::create(element::i64, {}, {2});

    bool with_fq = true;
    std::shared_ptr<ov::Node> gather_0;
    if (with_fq) {
        auto fq0 = std::make_shared<opset10::FakeQuantize>(transpose,
                                                           opset10::Constant::create(element::f32, Shape{}, {0}),
                                                           opset10::Constant::create(element::f32, Shape{}, {20}),
                                                           opset10::Constant::create(element::f32, Shape{}, {0}),
                                                           opset10::Constant::create(element::f32, Shape{}, {254}),
                                                           255);
        gather_0 = std::make_shared<ov::op::v1::Gather>(fq0, const_zero /*indices*/, const_zero /*axis*/);
    } else {
        gather_0 = std::make_shared<ov::op::v1::Gather>(transpose, const_zero /*indices*/, const_zero /*axis*/);
    }

    auto gather_1 = std::make_shared<ov::op::v1::Gather>(transpose, const_one /*indices*/, const_zero /*axis*/);
    auto gather_2 = std::make_shared<ov::op::v1::Gather>(transpose, const_two /*indices*/, const_zero /*axis*/);

    auto mul_const = ov::op::v0::Constant::create(precision, {1}, {2.0});
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(gather_1, mul_const);

    std::shared_ptr<ov::Node> mul2_fq = mul2;
    if (with_fq) {
        mul2_fq = std::make_shared<opset10::FakeQuantize>(mul2,
                                                          opset10::Constant::create(element::f32, Shape{}, {0}),
                                                          opset10::Constant::create(element::f32, Shape{}, {20}),
                                                          opset10::Constant::create(element::f32, Shape{}, {0}),
                                                          opset10::Constant::create(element::f32, Shape{}, {254}),
                                                          255);
    }

    auto mm_qk = std::make_shared<ov::op::v0::MatMul>(gather_0, mul2_fq, false, true);
    auto softmax = std::make_shared<ov::op::v1::Softmax>(mm_qk);
    auto mm_v = std::make_shared<ov::op::v0::MatMul>(softmax, gather_2, false, false);

#if 0 // Debug code
    functionRefs = std::make_shared<Model>(OutputVector{mm_v}, ParameterVector{param});
    ov::pass::Serialize serializer("matmul_gathers_ref.xml", "matmul_gathers_ref.bin");
    serializer.run_on_model(functionRefs);

    function = functionRefs->clone();
    ov::pass::Manager manager;
    // apply
    manager.register_pass<ov::pass::MatmulGatherDecomposition>();
    manager.run_passes(function);
    ov::pass::Serialize serializer2("matmul_gathers.xml", "matmul_gathers.bin");
    serializer2.run_on_model(function);
#else
    function = std::make_shared<ov::Model>(OutputVector{mm_v}, ParameterVector{param}, "MatMulGatherDecompose");
#endif
    abs_threshold = 1e-2f;
}

void MatMulGatherDecompose::check_results() {
    CheckNumberOfNodesWithType(compiledModel.get_runtime_model(), {"FullyConnected"}, 3);
}

std::string MatMulSplitDecompose::getTestCaseName(const testing::TestParamInfo<MatMulSplitDecomposeParams>& obj) {
    MatMulSplitDecomposeShapeParams shape_params;
    std::string device;
    std::tie(shape_params, device) = obj.param;
    std::ostringstream results;

    results << "input=" << shape_params.input_shape << "_";
    results << "weights=" << shape_params.weights_shape << "_";
    results << "transB=" << std::boolalpha << shape_params.trans_b << "_";
    results << "split_lengths_data=" << ov::test::utils::vec2str(shape_params.split_lengths_data) << "_";
    results << "reshape1=" << shape_params.reshape_shape1 << "_";
    results << "reshape2=" << shape_params.reshape_shape2 << "_";
    results << "reshape3=" << shape_params.reshape_shape3 << "_";
    results << "dev=" << device;
    return results.str();
}

void MatMulSplitDecompose::SetUp() {
    MatMulSplitDecomposeShapeParams shape_params;
    element::Type precision = element::f32;
    std::tie(shape_params, targetDevice) = GetParam();

    const auto& input_shape = shape_params.input_shape;
    const auto& weights_shape = shape_params.weights_shape;
    const auto& split_lengths_data = shape_params.split_lengths_data;
    const auto& reshape_shape1 = shape_params.reshape_shape1;
    const auto& reshape_shape2 = shape_params.reshape_shape2;
    const auto& reshape_shape3 = shape_params.reshape_shape3;

    auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);

    std::vector<float> weights_vals(shape_size(weights_shape), 2.0f);
    weights_vals = ov::test::utils::generate_float_numbers(shape_size(weights_shape), -0.1f, 0.1f);
    auto weights = ov::op::v0::Constant::create(precision, weights_shape, weights_vals);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, shape_params.trans_b);

    auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {input_shape.size() - 1});
    auto split_lengths = ov::op::v0::Constant::create(ov::element::i32, {split_lengths_data.size()}, split_lengths_data);
    auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(matmul, axis, split_lengths);

    auto shape1 = ov::op::v0::Constant::create(ov::element::i32, {reshape_shape1.size()}, reshape_shape1);
    auto shape2 = ov::op::v0::Constant::create(ov::element::i32, {reshape_shape2.size()}, reshape_shape2);
    auto shape3 = ov::op::v0::Constant::create(ov::element::i32, {reshape_shape3.size()}, reshape_shape3);
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(0), shape1, false);
    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(1), shape2, false);
    auto reshape3 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(2), shape3, false);

#if 0  // Debug code
    functionRefs = std::make_shared<Model>(OutputVector{reshape1, reshape2, reshape3}, ParameterVector{param});
    ov::pass::Serialize serializer("matmul_variadic_split_ref.xml", "matmul_variadic_split_ref.bin");
    serializer.run_on_model(functionRefs);

    function = functionRefs->clone();
    ov::pass::Manager manager;
    // apply
    manager.register_pass<ov::pass::MatmulVariadicSplitDecomposition>();
    manager.run_passes(function);
    ov::pass::Serialize serializer2("matmul_variadic_split.xml", "matmul_variadic_split.bin");
    serializer2.run_on_model(function);
#else
    function = std::make_shared<ov::Model>(OutputVector{reshape1, reshape2, reshape3},
                                           ParameterVector{param},
                                           "MatMulSplitDecompose");
#endif
    abs_threshold = 1e-4f;
}

void MatMulSplitDecompose::check_results() {
    CheckNumberOfNodesWithType(compiledModel.get_runtime_model(), {"FullyConnected"}, 3);
}

}  // namespace test
}  // namespace ov
