// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

using LogSoftmaxUnderflowParams = std::tuple<InputShape,          // input shape
                                             int64_t,             // axis
                                             ov::element::Type>;  // inference precision

// LogSoftmax of an f32 model whose inputs make x - max reach -128. exp(x - max) underflows there in f16 and f32, so
// computing log(softmax(x)) returns -inf where the reference is finite (Quartznet decoder logprobs on GPU FP16).
class LogSoftmaxUnderflowTest : public testing::WithParamInterface<LogSoftmaxUnderflowParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LogSoftmaxUnderflowParams>& obj) {
        const auto& [input_shape, axis, inference_precision] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({input_shape.first}) << "_TS=";
        for (const auto& shape : input_shape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "axis=" << axis << "_inference_precision=" << inference_precision.get_type_name();
        return result.str();
    }

protected:
    ov::element::Type inference_precision;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [input_shape, axis, precision] = GetParam();
        inference_precision = precision;
        init_input_shapes({input_shape});
        configuration[ov::hint::inference_precision.name()] = inference_precision.get_type_name();

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front());
        auto log_softmax = std::make_shared<ov::op::v5::LogSoftmax>(param, axis);
        auto result = std::make_shared<ov::op::v0::Result>(log_softmax);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "LogSoftmaxUnderflow");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto func_input = function->input(0);
        const ov::test::utils::InputGenerateData in_data(-64, 128, 8);
        inputs.insert({func_input.get_node_shared_ptr(),
                       ov::test::utils::create_and_fill_tensor(func_input.get_element_type(), targetInputStaticShapes.front(), in_data)});
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        const float rel_tolerance = inference_precision == ov::element::f16 ? 2e-3f : 1e-4f;
        for (size_t j = 0; j < expected.size(); ++j) {
            ASSERT_EQ(expected[j].get_size(), actual[j].get_size());
            const auto* expected_data = expected[j].data<const float>();
            const auto* actual_data = actual[j].data<const float>();
            const size_t size = expected[j].get_size();
            ASSERT_LT(*std::min_element(expected_data, expected_data + size), -104.f) << "inputs must reach the f32 exp underflow";
            for (size_t i = 0; i < size; ++i) {
                ASSERT_TRUE(std::isfinite(actual_data[i])) << "at index " << i << ", expected " << expected_data[i];
                ASSERT_NEAR(expected_data[i], actual_data[i], rel_tolerance * std::max(1.f, std::abs(expected_data[i]))) << "at index " << i;
            }
        }
    }
};

TEST_P(LogSoftmaxUnderflowTest, Inference) {
    run();
}

const std::vector<InputShape> input_shapes_last_axis = {
    {{}, {{1, 64, 29}}},
    {{-1, -1, 29}, {{1, 64, 29}, {2, 16, 29}}},
    {{}, {{2, 1000}}},
};

const std::vector<InputShape> input_shapes_feature_axis = {
    {{}, {{2, 48, 3, 2}}},
    {{-1, 48, -1, -1}, {{2, 48, 3, 2}, {1, 48, 1, 1}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmaxUnderflow_last_axis,
                         LogSoftmaxUnderflowTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_last_axis),
                                            ::testing::Values(-1),
                                            ::testing::Values(ov::element::f16, ov::element::f32)),
                         LogSoftmaxUnderflowTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmaxUnderflow_feature_axis,
                         LogSoftmaxUnderflowTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_feature_axis),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::element::f16, ov::element::f32)),
                         LogSoftmaxUnderflowTest::getTestCaseName);
}  // namespace
