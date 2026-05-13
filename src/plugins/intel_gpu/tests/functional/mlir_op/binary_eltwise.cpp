// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

// Params: lhs shape, rhs shape, precision
using BinaryElementwiseParams = std::tuple<ov::Shape, ov::Shape, ov::element::Type>;

template <typename Op>
class BinaryElementwiseTest : public testing::WithParamInterface<BinaryElementwiseParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BinaryElementwiseParams>& obj) {
        const auto& [lhs_shape, rhs_shape, precision] = obj.param;
        std::ostringstream result;
        result << "LHS=" << ov::test::utils::vec2str(lhs_shape) << "_";
        result << "RHS=" << ov::test::utils::vec2str(rhs_shape) << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [lhs_shape, rhs_shape, precision] = GetParam();
        auto lhs = std::make_shared<ov::op::v0::Parameter>(precision, lhs_shape);
        auto rhs = std::make_shared<ov::op::v0::Parameter>(precision, rhs_shape);
        auto op = std::make_shared<Op>(lhs, rhs);
        auto result = std::make_shared<ov::op::v0::Result>(op);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{lhs, rhs});
    }
};

using AddTest = BinaryElementwiseTest<ov::op::v1::Add>;
using SubtractTest = BinaryElementwiseTest<ov::op::v1::Subtract>;
using MultiplyTest = BinaryElementwiseTest<ov::op::v1::Multiply>;
using DivideTest = BinaryElementwiseTest<ov::op::v1::Divide>;
using PowerTest = BinaryElementwiseTest<ov::op::v1::Power>;
using MaximumTest = BinaryElementwiseTest<ov::op::v1::Maximum>;
using MinimumTest = BinaryElementwiseTest<ov::op::v1::Minimum>;
using FloorModTest = BinaryElementwiseTest<ov::op::v1::FloorMod>;
using ModTest = BinaryElementwiseTest<ov::op::v1::Mod>;

TEST_P(AddTest, Inference) {
    run();
}
TEST_P(SubtractTest, Inference) {
    run();
}
TEST_P(MultiplyTest, Inference) {
    run();
}
TEST_P(DivideTest, Inference) {
    run();
}
TEST_P(PowerTest, Inference) {
    run();
}
TEST_P(MaximumTest, Inference) {
    run();
}
TEST_P(MinimumTest, Inference) {
    run();
}
TEST_P(FloorModTest, Inference) {
    run();
}
TEST_P(ModTest, Inference) {
    run();
}

// Same-shape
const auto same_shape_params =
    ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 1536}), ::testing::Values(ov::Shape{1, 1024, 1536}), ::testing::Values(ov::element::f16));
// Broadcast: arg0=[1, 1024, 1536], arg1=[1536]
const auto broadcast_params =
    ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 1536}), ::testing::Values(ov::Shape{1536}), ::testing::Values(ov::element::f16));

#define INSTANTIATE_TS(Name)                                                                                             \
    INSTANTIATE_TEST_SUITE_P(mlir_BinaryElementwise##Name##_same_shape, Name, same_shape_params, Name::getTestCaseName); \
    INSTANTIATE_TEST_SUITE_P(mlir_BinaryElementwise##Name##_broadcast, Name, broadcast_params, Name::getTestCaseName)

INSTANTIATE_TS(AddTest);
INSTANTIATE_TS(SubtractTest);
INSTANTIATE_TS(MultiplyTest);
INSTANTIATE_TS(DivideTest);
INSTANTIATE_TS(PowerTest);
INSTANTIATE_TS(MaximumTest);
INSTANTIATE_TS(MinimumTest);
INSTANTIATE_TS(FloorModTest);
INSTANTIATE_TS(ModTest);

// Params: input shape, constant shape, precision
using BinaryElementwiseConstParams = std::tuple<ov::Shape, ov::Shape, ov::element::Type>;

template <typename Op>
class BinaryElementwiseConstTest : public testing::WithParamInterface<BinaryElementwiseConstParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BinaryElementwiseConstParams>& obj) {
        const auto& [input_shape, const_shape, precision] = obj.param;
        std::ostringstream result;
        result << "Input=" << ov::test::utils::vec2str(input_shape) << "_";
        result << "Const=" << ov::test::utils::vec2str(const_shape) << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [input_shape, const_shape, precision] = GetParam();
        ov::test::utils::InputGenerateData gen;
        gen.start_from = 3;
        gen.range = 1;
        auto const_tensor = ov::test::utils::create_and_fill_tensor(precision, const_shape, gen);
        auto input = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
        auto constant = std::make_shared<ov::op::v0::Constant>(const_tensor);
        auto op = std::make_shared<Op>(input, constant);
        auto result = std::make_shared<ov::op::v0::Result>(op);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
};

using AddConstTest = BinaryElementwiseConstTest<ov::op::v1::Add>;
using SubtractConstTest = BinaryElementwiseConstTest<ov::op::v1::Subtract>;
using MultiplyConstTest = BinaryElementwiseConstTest<ov::op::v1::Multiply>;
using DivideConstTest = BinaryElementwiseConstTest<ov::op::v1::Divide>;

TEST_P(AddConstTest, Inference) {
    run();
}
TEST_P(SubtractConstTest, Inference) {
    run();
}
TEST_P(MultiplyConstTest, Inference) {
    run();
}
TEST_P(DivideConstTest, Inference) {
    run();
}

// Scalar constant broadcast
const auto const_scalar_params =
    ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 1536}), ::testing::Values(ov::Shape{}), ::testing::Values(ov::element::f16));
// 1D constant broadcast
const auto const_broadcast_params =
    ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 1536}), ::testing::Values(ov::Shape{1536}), ::testing::Values(ov::element::f16));

#undef INSTANTIATE_TS
#define INSTANTIATE_TS(Name)                                                                                           \
    INSTANTIATE_TEST_SUITE_P(mlir_BinaryElementwise##Name##_scalar, Name, const_scalar_params, Name::getTestCaseName); \
    INSTANTIATE_TEST_SUITE_P(mlir_BinaryElementwise##Name##_broadcast, Name, const_broadcast_params, Name::getTestCaseName)

INSTANTIATE_TS(AddConstTest);
INSTANTIATE_TS(SubtractConstTest);
INSTANTIATE_TS(MultiplyConstTest);
INSTANTIATE_TS(DivideConstTest);

}  // namespace
