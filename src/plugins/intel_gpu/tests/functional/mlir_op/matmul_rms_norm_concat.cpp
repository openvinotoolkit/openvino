// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/transpose.hpp"
#include "shared_test_classes/base/benchmark.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

//    A(1xSEQx1536xf16)
//    ▼
// MatMul(transpose B) → Add(const) → Reshape(1xSEQx24x64) → Transpose(1x24xSEQx64)
//    ▲                                                      ▼             ▼
//    B(1536x1536xf16)                               Power(B=const)        │
//                                                           ▼             │
//                                                 ReduceMean(axis=3)      │
//                                                           ▼             │
//                                                    Add(B=const)         │
//                                                           ▼             │
//                                                          Sqrt           │
//                                                           ▼             │
//                                                 Divide(B=const)         │
//                                                           ▼             ▼
//                                                         Multiply(Divide × Transpose)
//                                                                  ▼
//                                                               Multiply(× const)

// Builds the MatMul+RmsNorm subgraph for a given A shape and shared B parameter.
// Returns the output node (Multiply × scale).
static std::shared_ptr<ov::Node> build_matmul_rmsnorm(ov::element::Type prec,
                                                      const std::shared_ptr<ov::op::v0::Parameter>& param_a,
                                                      const std::shared_ptr<ov::op::v0::Parameter>& param_b) {
    const auto& a_shape = param_a->get_shape();
    const int64_t hidden = static_cast<int64_t>(a_shape.back());
    const int64_t seq = static_cast<int64_t>(a_shape[1]);
    const int64_t heads = 24;
    const int64_t head_size = hidden / heads;

    auto matmul = std::make_shared<ov::op::v0::MatMul>(param_a, param_b, false, true);

    auto bias = ov::op::v0::Constant::create(prec, {(size_t)hidden}, std::vector<float>(hidden, 0.1f));
    auto add1 = std::make_shared<ov::op::v1::Add>(matmul, bias);

    auto shape_val = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, seq, heads, head_size});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(add1, shape_val, false);

    auto order = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, order);

    auto exp2 = ov::op::v0::Constant::create(prec, {1}, std::vector<float>{2.f});
    auto power = std::make_shared<ov::op::v1::Power>(transpose, exp2);

    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{3});
    auto reduce_mean = std::make_shared<ov::op::v1::ReduceMean>(power, axes, true);

    auto eps = ov::op::v0::Constant::create(prec, {1}, std::vector<float>{1e-5f});
    auto add2 = std::make_shared<ov::op::v1::Add>(reduce_mean, eps);

    auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(add2);

    auto one = ov::op::v0::Constant::create(prec, {1}, std::vector<float>{1.f});
    auto divide = std::make_shared<ov::op::v1::Divide>(one, sqrt_node);

    auto mul1 = std::make_shared<ov::op::v1::Multiply>(divide, transpose);

    auto scale = ov::op::v0::Constant::create(prec, {1}, std::vector<float>{2.f});
    return std::make_shared<ov::op::v1::Multiply>(mul1, scale);
}

// ── MatMulRmsnormTest ─────────────────────────────────────────────────────────

using MatMulRmsnormParams = std::tuple<ov::Shape,   // A shape
                                       ov::Shape>;  // B shape

class MatMulRmsnormTest : public testing::WithParamInterface<MatMulRmsnormParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulRmsnormParams>& obj) {
        const auto& [a_shape, b_shape] = obj.param;
        std::ostringstream result;
        result << "A=" << ov::test::utils::vec2str(a_shape) << "_";
        result << "B=" << ov::test::utils::vec2str(b_shape);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto prec = ov::element::f16;
        const auto& [a_shape, b_shape] = GetParam();

        auto param_a = std::make_shared<ov::op::v0::Parameter>(prec, a_shape);
        auto param_b = std::make_shared<ov::op::v0::Parameter>(prec, b_shape);
        auto out = build_matmul_rmsnorm(prec, param_a, param_b);
        auto result = std::make_shared<ov::op::v0::Result>(out);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_a, param_b}, "MatMulRmsnorm");
    }
};

class MatMulRmsnormBenchmark : public ov::test::BenchmarkLayerTest<MatMulRmsnormTest> {};

TEST_P(MatMulRmsnormTest, Inference) {
    run();
}
TEST_P(MatMulRmsnormBenchmark, Inference) {
    run_benchmark("MLIROp");
}

const auto rmsnormParams = ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 1536}), ::testing::Values(ov::Shape{1536, 1536}));
INSTANTIATE_TEST_SUITE_P(mlir_MatMulRmsnormTest, MatMulRmsnormTest, rmsnormParams, MatMulRmsnormTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(bench_MatMulRmsnormBenchmark, MatMulRmsnormBenchmark, rmsnormParams, MatMulRmsnormTest::getTestCaseName);

// ── MatMulRmsnormConcatTest ───────────────────────────────────────────────────
//
// Runs two MatMul+RmsNorm graphs (different A sequence lengths) sharing B,
// then concatenates their outputs along the sequence axis (axis=2).
//
//   branch0: A0(1xSEQ0x1536) ─┐
//   branch1: A1(1xSEQ1x1536) ─┤ → Concat(axis=2)
//                              └─ shared B(1536x1536)

using MatMulRmsnormConcatParams = std::tuple<ov::Shape,   // A0 shape
                                             ov::Shape,   // A1 shape
                                             ov::Shape>;  // B shape

class MatMulRmsnormConcatTest : public testing::WithParamInterface<MatMulRmsnormConcatParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulRmsnormConcatParams>& obj) {
        const auto& [a0_shape, a1_shape, b_shape] = obj.param;
        std::ostringstream result;
        result << "A0=" << ov::test::utils::vec2str(a0_shape) << "_";
        result << "A1=" << ov::test::utils::vec2str(a1_shape) << "_";
        result << "B=" << ov::test::utils::vec2str(b_shape);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto prec = ov::element::f16;
        const auto& [a0_shape, a1_shape, b_shape] = GetParam();

        auto param_a0 = std::make_shared<ov::op::v0::Parameter>(prec, a0_shape);
        auto param_a1 = std::make_shared<ov::op::v0::Parameter>(prec, a1_shape);
        auto param_b = std::make_shared<ov::op::v0::Parameter>(prec, b_shape);

        auto out0 = build_matmul_rmsnorm(prec, param_a0, param_b);
        auto out1 = build_matmul_rmsnorm(prec, param_a1, param_b);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{out0, out1}, 2);
        auto result = std::make_shared<ov::op::v0::Result>(concat);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_a0, param_a1, param_b}, "MatMulRmsnormConcat");
    }
};

class MatMulRmsnormConcatBenchmark : public ov::test::BenchmarkLayerTest<MatMulRmsnormConcatTest> {};

TEST_P(MatMulRmsnormConcatTest, Inference) {
    run();
}
TEST_P(MatMulRmsnormConcatBenchmark, Inference) {
    run_benchmark("MLIROp");
}

const auto concatParams =
    ::testing::Combine(::testing::Values(ov::Shape{1, 128, 1536}), ::testing::Values(ov::Shape{1, 1024, 1536}), ::testing::Values(ov::Shape{1536, 1536}));
INSTANTIATE_TEST_SUITE_P(mlir_MatMulRmsnormConcatTest, MatMulRmsnormConcatTest, concatParams, MatMulRmsnormConcatTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(bench_MatMulRmsnormConcatBenchmark, MatMulRmsnormConcatBenchmark, concatParams, MatMulRmsnormConcatTest::getTestCaseName);
}  // namespace
