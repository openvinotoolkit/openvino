// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/minimum_maximum.hpp"
#include "common_test_utils/test_constants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
using ov::test::MaxMinLayerTest;
using ov::test::utils::InputLayerType;
using ov::test::utils::MinMaxOpType;

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
        {{2}, {1}},
        {{1, 1, 1, 3}, {1}},
        {{1, 2, 4}, {1}},
        {{1, 4, 4}, {1}},
        {{1, 4, 4, 1}, {1}},
        {{256, 56}, {256, 56}},
        {{8, 1, 6, 1}, {7, 1, 5}},
};

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
};

const std::vector<MinMaxOpType> op_types = {
        MinMaxOpType::MINIMUM,
        MinMaxOpType::MAXIMUM,
};

const std::vector<InputLayerType> second_input_types = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

class MaxMinNaNLayerTest : public MaxMinLayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        const auto& params = GetParam();
        const auto op_type = std::get<1>(params);
        const auto model_type = std::get<2>(params);
        const auto second_input_type = std::get<3>(params);
        ASSERT_EQ(op_type, MinMaxOpType::MINIMUM);
        ASSERT_EQ(model_type, ov::element::f32);
        ASSERT_EQ(second_input_type, InputLayerType::PARAMETER);
        ASSERT_EQ(target_input_static_shapes.size(), 2);
        ASSERT_EQ(ov::shape_size(target_input_static_shapes[0]), ov::shape_size(target_input_static_shapes[1]));

        inputs.clear();
        const std::vector<std::vector<float>> input_values = {
            {std::numeric_limits<float>::quiet_NaN(), 5.F, std::numeric_limits<float>::quiet_NaN(), -2.F},
            {3.F, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), -4.F},
        };

        const auto& func_inputs = function->inputs();
        for (size_t input_idx = 0; input_idx < func_inputs.size(); ++input_idx) {
            ov::Tensor tensor{ov::element::f32, target_input_static_shapes[input_idx]};
            auto* data = tensor.data<float>();
            const auto tensor_size = tensor.get_size();
            for (size_t i = 0; i < tensor_size; ++i) {
                data[i] = input_values[input_idx][i % input_values[input_idx].size()];
            }
            inputs.insert({func_inputs[input_idx].get_node_shared_ptr(), tensor});
        }

        expected_output = ov::Tensor{ov::element::f32, target_input_static_shapes[0]};
        const auto* lhs = inputs.at(func_inputs[0].get_node_shared_ptr()).data<const float>();
        const auto* rhs = inputs.at(func_inputs[1].get_node_shared_ptr()).data<const float>();
        auto* expected = expected_output.data<float>();
        for (size_t i = 0; i < expected_output.get_size(); ++i) {
            if (std::isnan(lhs[i])) {
                expected[i] = lhs[i];
            } else if (std::isnan(rhs[i])) {
                expected[i] = rhs[i];
            } else {
                expected[i] = std::min(lhs[i], rhs[i]);
            }
        }
    }

    std::vector<ov::Tensor> calculate_refs() override {
        return {expected_output};
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), 1);
        ASSERT_EQ(actual.size(), 1);
        ASSERT_EQ(expected[0].get_shape(), actual[0].get_shape());

        const auto* expected_data = expected[0].data<const float>();
        const auto* actual_data = actual[0].data<const float>();
        for (size_t i = 0; i < expected[0].get_size(); ++i) {
            if (std::isnan(expected_data[i])) {
                EXPECT_TRUE(std::isnan(actual_data[i])) << "at index " << i;
            } else {
                EXPECT_EQ(expected_data[i], actual_data[i]) << "at index " << i;
            }
        }
    }

private:
    ov::Tensor expected_output;
};

const std::vector<std::vector<ov::Shape>> nan_input_shapes_static = {
        {{4}, {4}},
};

TEST_P(MaxMinNaNLayerTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_maximum, MaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                ::testing::ValuesIn(op_types),
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(second_input_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        MaxMinLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_minimum_nan, MaxMinNaNLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(nan_input_shapes_static)),
                                ::testing::Values(MinMaxOpType::MINIMUM),
                                ::testing::Values(ov::element::f32),
                                ::testing::Values(InputLayerType::PARAMETER),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        MaxMinNaNLayerTest::getTestCaseName);

}  // namespace
