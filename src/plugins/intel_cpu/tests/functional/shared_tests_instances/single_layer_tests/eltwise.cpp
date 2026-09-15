// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "single_op_tests/eltwise.hpp"

namespace {
using ov::test::EltwiseLayerTest;
using ov::test::utils::InputLayerType;
using ov::test::utils::OpType;
using ov::test::utils::EltwiseTypes;

class Int64PowerLayerCPUTest : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const ov::Shape baseShape{2, 10};
        const ov::Shape exponentShape{1, 10};
        auto base = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, baseShape);
        auto exponent = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, exponentShape);
        auto power = std::make_shared<ov::op::v1::Power>(base, exponent);
        function = std::make_shared<ov::Model>(ov::OutputVector{power}, ov::ParameterVector{base, exponent});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& functionInputs = function->inputs();

        ov::Tensor baseTensor(ov::element::i64, targetInputStaticShapes[0]);
        ov::Tensor exponentTensor(ov::element::i64, targetInputStaticShapes[1]);

        constexpr std::array<int64_t, 20> bases = {
            13,  17,  19,  23,  -2, -3, 5,  7,  2, -2,
            -13, -17, -19, -23, 2,  3,  1, -1, 1, -1,
        };
        constexpr std::array<int64_t, 10> exponents = {8, 7, 7, 6, 3, 2, 1, 0, -1, -2};
        std::copy(bases.begin(), bases.end(), baseTensor.data<int64_t>());
        std::copy(exponents.begin(), exponents.end(), exponentTensor.data<int64_t>());

        inputs.insert({functionInputs[0].get_node_shared_ptr(), baseTensor});
        inputs.insert({functionInputs[1].get_node_shared_ptr(), exponentTensor});
    }
};

TEST_F(Int64PowerLayerCPUTest, CompareWithRefs) {
    run();
}

class Int64PowerFusedEltwiseLayerCPUTest : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const ov::Shape inputShape{4};
        auto base = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputShape);
        auto exponent = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputShape);
        auto power = std::make_shared<ov::op::v1::Power>(base, exponent);
        auto addValue = ov::op::v0::Constant::create(ov::element::i64, inputShape, {1, 2, 3, 4});
        auto add = std::make_shared<ov::op::v1::Add>(power, addValue);
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{base, exponent});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& functionInputs = function->inputs();

        ov::Tensor baseTensor(ov::element::i64, targetInputStaticShapes[0]);
        ov::Tensor exponentTensor(ov::element::i64, targetInputStaticShapes[1]);

        constexpr std::array<int64_t, 4> bases = {13, 17, 19, 23};
        constexpr std::array<int64_t, 4> exponents = {8, 7, 7, 6};
        std::copy(bases.begin(), bases.end(), baseTensor.data<int64_t>());
        std::copy(exponents.begin(), exponents.end(), exponentTensor.data<int64_t>());

        inputs.insert({functionInputs[0].get_node_shared_ptr(), baseTensor});
        inputs.insert({functionInputs[1].get_node_shared_ptr(), exponentTensor});
    }
};

TEST_F(Int64PowerFusedEltwiseLayerCPUTest, CompareWithRefs) {
    run();
}

class Int64PowerFusedChildLayerCPUTest : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const ov::Shape inputShape{4};
        auto base = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputShape);
        auto exponent = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputShape);
        auto addValue = ov::op::v0::Constant::create(ov::element::i64, inputShape, {1, 2, 3, 4});
        auto add = std::make_shared<ov::op::v1::Add>(base, addValue);
        auto power = std::make_shared<ov::op::v1::Power>(add, exponent);
        function = std::make_shared<ov::Model>(ov::OutputVector{power}, ov::ParameterVector{base, exponent});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& functionInputs = function->inputs();

        ov::Tensor baseTensor(ov::element::i64, targetInputStaticShapes[0]);
        ov::Tensor exponentTensor(ov::element::i64, targetInputStaticShapes[1]);

        constexpr std::array<int64_t, 4> bases = {12, 15, 16, 19};
        constexpr std::array<int64_t, 4> exponents = {8, 7, 7, 6};
        std::copy(bases.begin(), bases.end(), baseTensor.data<int64_t>());
        std::copy(exponents.begin(), exponents.end(), exponentTensor.data<int64_t>());

        inputs.insert({functionInputs[0].get_node_shared_ptr(), baseTensor});
        inputs.insert({functionInputs[1].get_node_shared_ptr(), exponentTensor});
    }
};

TEST_F(Int64PowerFusedChildLayerCPUTest, CompareWithRefs) {
    run();
}

std::vector<std::vector<ov::Shape>> in_shapes_static = {
        {{2}},
        {{2, 200}},
        {{10, 200}},
        {{1, 10, 100}},
        {{4, 4, 16}},
        {{1, 1, 1, 3}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{1, 4, 4, 1}},
        {{16, 16, 16, 16, 16}},
        {{16, 16, 16, 16, 1}},
        {{16, 16, 16, 1, 16}},
        {{16, 32, 1, 1, 1}},
        {{1, 1, 1, 1, 1, 1, 3}},
        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
};

std::vector<std::vector<ov::Shape>> in_shapes_static_check_collapse = {
        {{16, 16, 16, 16}, {16, 16, 16, 1}},
        {{16, 16, 16, 1}, {16, 16, 16, 1}},
        {{16, 16, 16, 16}, {16, 16, 1, 16}},
        {{16, 16, 1, 16}, {16, 16, 1, 16}},
};

std::vector<std::vector<ov::test::InputShape>> in_shapes_dynamic = {
        {{{ov::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}},
         {{ov::Dimension(1, 10), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<std::vector<ov::test::InputShape>> in_shapes_dynamic_large_upper_bound = {
        {{{ov::Dimension(1, 1000000000000), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<ov::test::ElementType> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

std::vector<InputLayerType> secondary_input_types = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

std::vector<InputLayerType> secondary_input_types_dynamic = {
        InputLayerType::PARAMETER,
};

std::vector<OpType> op_types = {
        OpType::SCALAR,
        OpType::VECTOR,
};

std::vector<OpType> op_types_dynamic = {
        OpType::VECTOR,
};

std::vector<EltwiseTypes> eltwise_op_types = {
        EltwiseTypes::ADD,
        EltwiseTypes::MULTIPLY,
        EltwiseTypes::SUBTRACT,
        EltwiseTypes::DIVIDE,
        EltwiseTypes::FLOOR_MOD,
        EltwiseTypes::SQUARED_DIFF,
        EltwiseTypes::POWER,
        EltwiseTypes::MOD
};

std::vector<EltwiseTypes> eltwise_op_types_dynamic = {
        EltwiseTypes::ADD,
        EltwiseTypes::MULTIPLY,
        EltwiseTypes::SUBTRACT,
        EltwiseTypes::POWER,
};

std::vector<EltwiseTypes> bitshift_types = {
        EltwiseTypes::LEFT_SHIFT,
        EltwiseTypes::RIGHT_SHIFT
};

ov::test::Config additional_config = {};

const auto multiply_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_static)),
                       ::testing::ValuesIn(eltwise_op_types),
                       ::testing::ValuesIn(secondary_input_types),
                       ::testing::ValuesIn(op_types),
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(additional_config));

const auto collapsing_params = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_static_check_collapse)),
    ::testing::ValuesIn(eltwise_op_types),
    ::testing::ValuesIn(secondary_input_types),
    ::testing::Values(op_types[1]),
    ::testing::ValuesIn(model_types),
    ::testing::Values(ov::element::dynamic),
    ::testing::Values(ov::element::dynamic),
    ::testing::Values(ov::test::utils::DEVICE_CPU),
    ::testing::Values(additional_config));

const auto multiply_params_dynamic = ::testing::Combine(::testing::ValuesIn(in_shapes_dynamic),
                                                        ::testing::ValuesIn(eltwise_op_types_dynamic),
                                                        ::testing::ValuesIn(secondary_input_types_dynamic),
                                                        ::testing::ValuesIn(op_types_dynamic),
                                                        ::testing::ValuesIn(model_types),
                                                        ::testing::Values(ov::element::dynamic),
                                                        ::testing::Values(ov::element::dynamic),
                                                        ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                        ::testing::Values(additional_config));

const auto multiply_params_dynamic_large_upper_bound =
    ::testing::Combine(::testing::ValuesIn(in_shapes_dynamic_large_upper_bound),
                       ::testing::Values(EltwiseTypes::ADD),
                       ::testing::ValuesIn(secondary_input_types_dynamic),
                       ::testing::ValuesIn(op_types_dynamic),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_static, EltwiseLayerTest, multiply_params, EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_static_check_collapsing, EltwiseLayerTest, collapsing_params, EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, EltwiseLayerTest, multiply_params_dynamic, EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic_large_upper_bound,
                         EltwiseLayerTest,
                         multiply_params_dynamic_large_upper_bound,
                         EltwiseLayerTest::getTestCaseName);


std::vector<std::vector<ov::Shape>> inShapesSingleThread = {
        {{1, 1, 1, 2}},
        {{1, 1, 1, 4}},
        {{1, 2, 3, 4}},
        {{2, 2, 2, 2}},
        {{2, 1, 2, 1, 2, 2}},
};

std::vector<EltwiseTypes> eltwise_op_typesSingleThread = {
        EltwiseTypes::ADD,
        EltwiseTypes::MULTIPLY,
        EltwiseTypes::POWER,
};

ov::AnyMap additional_config_single_thread = {
    ov::inference_num_threads(1),
};

const auto single_thread_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesSingleThread)),
                       ::testing::ValuesIn(eltwise_op_typesSingleThread),
                       ::testing::ValuesIn(secondary_input_types),
                       ::testing::ValuesIn(op_types),
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(additional_config_single_thread));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThread, EltwiseLayerTest, single_thread_params, EltwiseLayerTest::getTestCaseName);

std::vector<ov::test::ElementType> intOnly_netPrecisions = {
        ov::element::i32,
        ov::element::i8,
        ov::element::u8,
        ov::element::u16,
        ov::element::i16,
        ov::element::u32,
};

std::vector<std::vector<ov::Shape>> in_shapes_static_small_set = {
        {{2}},
        {{2, 10}, {1}},
        {{4, 3, 8}, {1, 8}},
        {{2, 7, 5, 4}, {1, 7, 1, 1}},
        {{1, 7, 5, 1}, {2, 7, 1, 4}},
};

const auto bitwise_shift_params_static =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_static_small_set)),
                       ::testing::ValuesIn(bitshift_types),
                       ::testing::ValuesIn(secondary_input_types),
                       ::testing::ValuesIn(op_types_dynamic),
                       ::testing::ValuesIn(intOnly_netPrecisions),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_shared_CompareWithRefs_BitwiseShift_Static,
                         EltwiseLayerTest,
                         bitwise_shift_params_static,
                         EltwiseLayerTest::getTestCaseName);

const auto bitwise_shift_params_dynamic = ::testing::Combine(::testing::ValuesIn(in_shapes_dynamic),
                                                             ::testing::ValuesIn(bitshift_types),
                                                             ::testing::ValuesIn(secondary_input_types_dynamic),
                                                             ::testing::ValuesIn(op_types_dynamic),
                                                             ::testing::ValuesIn(intOnly_netPrecisions),
                                                             ::testing::Values(ov::element::dynamic),
                                                             ::testing::Values(ov::element::dynamic),
                                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                             ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_shared_CompareWithRefs_BitwiseShift_Dynamic,
                         EltwiseLayerTest,
                         bitwise_shift_params_dynamic,
                         EltwiseLayerTest::getTestCaseName);

} // namespace
