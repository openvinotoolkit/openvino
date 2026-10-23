// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/normalize_l2.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/runtime/exec_model_info.hpp"

namespace {
using ov::test::NormalizeL2LayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<std::vector<int64_t>> axes = {
        {},
        {1},
};
const std::vector<float> eps = {1e-7f, 1e-6f, 1e-5f, 1e-4f};

const std::vector<ov::op::EpsMode> epsMode = {
        ov::op::EpsMode::ADD,
        ov::op::EpsMode::MAX,
};

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2,
                         NormalizeL2LayerTest,
                         testing::Combine(testing::ValuesIn(axes),
                                          testing::ValuesIn(eps),
                                          testing::ValuesIn(epsMode),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{1, 3, 10, 5}})),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         NormalizeL2LayerTest::getTestCaseName);

using NormalizeAcrossXParams = std::tuple<ov::Shape, ov::element::Type, bool>;

class NormalizeL2AcrossX : public testing::TestWithParam<NormalizeAcrossXParams> {};

TEST_P(NormalizeL2AcrossX, OnnxPatternUsesSingleKernel) {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        const auto& [shape, type, decomposed_pattern] = GetParam();
        const auto input = std::make_shared<ov::opset13::Parameter>(type, ov::PartialShape(shape));
        const auto axes_node = ov::opset13::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        const float epsilon = std::numeric_limits<float>::epsilon();
        std::shared_ptr<ov::Node> result;
        if (decomposed_pattern) {
                const auto exponent = ov::opset13::Constant::create(type, ov::Shape{}, {2});
                const auto square = std::make_shared<ov::opset13::Power>(input, exponent);
                const auto sum = std::make_shared<ov::opset13::ReduceSum>(square, axes_node, true);
                const auto eps_node = ov::opset13::Constant::create(type, ov::Shape{}, {epsilon});
                const auto biased_sum = std::make_shared<ov::opset13::Add>(sum, eps_node);
                result = std::make_shared<ov::opset13::Divide>(input, std::make_shared<ov::opset13::Sqrt>(biased_sum));
        } else {
                result = std::make_shared<ov::opset13::NormalizeL2>(input, axes_node, epsilon, ov::op::EpsMode::ADD);
        }
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input});
        ov::Core core;
        auto compiled = core.compile_model(model,
                                                                                         ov::test::utils::DEVICE_GPU,
                                                                                         {ov::hint::inference_precision(type)});

        auto request = compiled.create_infer_request();
        auto tensor = request.get_input_tensor();
        const auto width = shape.back();
        std::vector<float> values(tensor.get_size());
        for (size_t index = 0; index < values.size(); ++index) {
                const auto row = index / width;
                float value = row == 0 ? 0.0f : row == 1 ? 1e-4f : static_cast<float>(static_cast<int>(index % 23) - 11);
                if (type == ov::element::f16) {
                        tensor.data<ov::float16>()[index] = ov::float16(value);
                        value = static_cast<float>(tensor.data<ov::float16>()[index]);
                } else {
                        tensor.data<float>()[index] = value;
                }
                values[index] = value;
        }
        request.infer();
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_shape(), shape);
        for (size_t row = 0; row < values.size() / width; ++row) {
                double sum = epsilon;
                for (size_t column = 0; column < width; ++column) {
                        const auto value = values[row * width + column];
                        sum += static_cast<double>(value) * value;
                }
                for (size_t column = 0; column < width; ++column) {
                        const auto index = row * width + column;
                        const float actual = type == ov::element::f16 ? static_cast<float>(output.data<const ov::float16>()[index])
                                                                                                                 : output.data<const float>()[index];
                        ASSERT_NEAR(actual, values[index] / std::sqrt(sum), type == ov::element::f16 ? 0.002f : 2e-6f)
                                << "index=" << index;
                }
        }
}

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2AcrossX,
                                                 NormalizeL2AcrossX,
                                                 testing::Combine(testing::Values(ov::Shape{1, 512, 5120},
                                                                                                                  ov::Shape{2, 3, 17},
                                                                                                                  ov::Shape{2, 3, 2, 17},
                                                                                                                  ov::Shape{1, 2, 2, 1}),
                                                                                  testing::Values(ov::element::f32, ov::element::f16),
                                                                                  testing::Bool()));

TEST(NormalizeL2AcrossXFallback, UnsupportedAxesAndMaxEpsilon) {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        ov::Core core;
        for (const auto& reduction_axes : {std::vector<int64_t>{0, 2}, std::vector<int64_t>{2}}) {
                const auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f32, ov::PartialShape{2, 3, 17});
                const auto axes_node = ov::opset13::Constant::create(ov::element::i64,
                                                                                                                   ov::Shape{reduction_axes.size()},
                                                                                                                   reduction_axes);
                const auto norm = std::make_shared<ov::opset13::NormalizeL2>(input, axes_node, 1e-7f, ov::op::EpsMode::MAX);
                const auto model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{input});
                const auto compiled = core.compile_model(model,
                                                                                                 ov::test::utils::DEVICE_GPU,
                                                                                                 {ov::hint::inference_precision(ov::element::f32)});
                for (const auto& node : compiled.get_runtime_model()->get_ops()) {
                        const auto& info = node->get_rt_info();
                        const auto impl = info.find(ov::exec_model_info::IMPL_TYPE);
                        if (impl != info.end()) {
                                ASSERT_EQ(impl->second.as<std::string>().find("normalize_gpu_across_x"), std::string::npos);
                        }
                }
        }
}
}  // namespace
