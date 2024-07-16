// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_split_decompose.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "transformations/common_optimizations/matmul_split_decomposition.hpp"

namespace ov {
namespace test {

std::string MatMulGatherDecompose::getTestCaseName(const testing::TestParamInfo<MatMulGatherDecomposeParams>& obj) {
    MatMulGatherDecomposeShapeParams shape_params;
    std::string device;
    bool enable_fq;
    std::tie(shape_params, device, enable_fq) = obj.param;
    std::ostringstream results;

    results << "input=" << shape_params.input_shape << "_";
    results << "weights=" << shape_params.weights_shape << "_";
    results << "trans_b=" << std::boolalpha << shape_params.trans_b << "_";
    results << "have_bias=" << std::boolalpha << shape_params.have_bias << "_";
    if (shape_params.have_bias) {
        results << "bias=" << shape_params.bias_shape << "_";
    }
    results << "reshape=" << shape_params.reshape_shape << "_";
    results << "dev=" << device << "_";
    results << "enable_fq=" << enable_fq;

    return results.str();
}

void MatMulGatherDecompose::SetUp() {
    MatMulGatherDecomposeShapeParams shape_params;
    element::Type precision = element::f32;
    bool enable_fq = false;
    std::tie(shape_params, targetDevice, enable_fq) = GetParam();

    const auto& input_shape = shape_params.input_shape;
    const auto& weights_shape = shape_params.weights_shape;
    const auto& have_bias = shape_params.have_bias;
    const auto& bias_shape = shape_params.bias_shape;
    const auto& reshape_shape = shape_params.reshape_shape;

    auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
    auto weights =
        ov::test::utils::make_constant(precision, weights_shape, ov::test::utils::InputGenerateData(-1, 1, 1000));
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, shape_params.trans_b);

    auto reshape_productor = std::shared_ptr<ov::Node>(matmul);
    if (have_bias) {
        auto bias =
            ov::test::utils::make_constant(precision, bias_shape, ov::test::utils::InputGenerateData(-1, 1, 1000));
        auto add = std::make_shared<ov::op::v1::Add>(matmul, bias);
        reshape_productor = std::shared_ptr<ov::Node>(add);
    }

    std::vector<int64_t> reshape_shape_vec;
    reshape_shape_vec.resize(reshape_shape.size());
    for (size_t i = 0; i < reshape_shape.size(); i++) {
        reshape_shape_vec[i] = static_cast<int64_t>(reshape_shape[i]);
    }

    auto reshape_const = ov::op::v0::Constant::create(element::i64, {reshape_shape_vec.size()}, reshape_shape_vec.data());
    auto reshape = std::make_shared<ov::op::v1::Reshape>(reshape_productor, reshape_const, false);

    auto transpose_const = ov::op::v0::Constant::create(element::i64, {5}, {2, 0, 3, 1, 4});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);

    auto const_zero = ov::op::v0::Constant::create(element::i64, {}, {0});
    auto const_one = ov::op::v0::Constant::create(element::i64, {}, {1});
    auto const_two = ov::op::v0::Constant::create(element::i64, {}, {2});

    std::shared_ptr<ov::Node> gather_0;
    if (enable_fq) {
        auto fq0 = std::make_shared<opset10::FakeQuantize>(transpose,
                                                           opset10::Constant::create(element::f32, Shape{}, {0}),
                                                           opset10::Constant::create(element::f32, Shape{}, {20}),
                                                           opset10::Constant::create(element::f32, Shape{}, {0}),
                                                           opset10::Constant::create(element::f32, Shape{}, {254}),
                                                           255);
        gather_0 = std::make_shared<ov::op::v8::Gather>(fq0, const_zero, const_zero);
    } else {
        gather_0 = std::make_shared<ov::op::v8::Gather>(transpose, const_zero, const_zero);
    }

    auto gather_1 = std::make_shared<ov::op::v8::Gather>(transpose, const_one, const_zero);
    auto gather_2 = std::make_shared<ov::op::v8::Gather>(transpose, const_two, const_zero);

    auto mul_const = ov::op::v0::Constant::create(precision, {1}, {2.0});
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(gather_1, mul_const);

    std::shared_ptr<ov::Node> mul2_fq = mul2;
    if (enable_fq) {
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

    function = std::make_shared<ov::Model>(OutputVector{mm_v}, ParameterVector{param}, "MatMulGatherDecompose");
    abs_threshold = 1e-4;
}

void MatMulGatherDecompose::check_results() {
    CheckNumberOfNodesWithType(compiledModel, {"FullyConnected"}, 3);
}

}  // namespace test
}  // namespace ov
