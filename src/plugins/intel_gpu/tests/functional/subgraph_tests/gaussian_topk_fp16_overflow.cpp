// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"

namespace {
// relu(x - (mean(x) + k * std(x))) with outliers whose squared deviation from the row mean
// exceeds the FP16 range; without FP32 threshold computation every row collapses to zero.
class GaussianTopKFP16Overflow : virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        abs_threshold = 0.5f;
        rel_threshold = 0.01f;

        const ov::Shape shape{1, 4, 1024};
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(x, axes, true);
        auto deviation = std::make_shared<ov::op::v1::Subtract>(x, mean);
        auto squared_deviation = std::make_shared<ov::op::v1::Multiply>(deviation, deviation);
        auto variance = std::make_shared<ov::op::v1::ReduceMean>(squared_deviation, axes, true);
        auto std_dev = std::make_shared<ov::op::v0::Sqrt>(variance);
        auto k = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.6448536f});
        auto threshold = std::make_shared<ov::op::v1::Add>(mean, std::make_shared<ov::op::v1::Multiply>(std_dev, k));
        auto relu = std::make_shared<ov::op::v0::Relu>(std::make_shared<ov::op::v1::Subtract>(x, threshold));
        function = std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{x});
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& param = function->get_parameters().front();
        auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, target_input_static_shapes.front(), 20, -10);
        auto* data = tensor.data<float>();
        const size_t row = target_input_static_shapes.front().back();
        for (size_t i = 0; i < tensor.get_size(); i += row) {
            data[i + 7] = 480.f;
            data[i + 300] = 450.f;
            data[i + 801] = -470.f;
        }
        inputs.insert({param, tensor});
    }
};

TEST_F(GaussianTopKFP16Overflow, Inference) {
    run();
}
}  // namespace
