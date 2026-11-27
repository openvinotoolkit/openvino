// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test::gpu {

inline std::vector<InputShape> runtime_weights_input_shapes(bool matmul) {
    const ov::Shape first_shape = matmul ? ov::Shape{1, 2, 3, 6} : ov::Shape{1, 6, 2, 3};
    const ov::Shape next_shape = matmul ? ov::Shape{2, 3, 2, 6} : ov::Shape{2, 6, 3, 2};
    const ov::PartialShape dynamic_shape = matmul ? ov::PartialShape{-1, -1, -1, 6} : ov::PartialShape{-1, 6, -1, -1};
    return {{ov::PartialShape(first_shape), {first_shape, first_shape, first_shape}},
            {dynamic_shape, {first_shape, first_shape, first_shape}},
            {dynamic_shape, {first_shape, next_shape, next_shape}}};
}

class RuntimeWeightsUpdateTestBase : public ov::test::SubgraphBaseTest {
protected:
    void setup_model(bool matmul, bool compressed, bool constant_weights, const InputShape& input_shape) {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration[ov::hint::inference_precision.name()] = ov::element::f16;
        abs_threshold = 0.001;
        rel_threshold = 0.001;
        const size_t channels = 6;
        const size_t outputs = 8;
        const ov::Shape weight_shape = matmul ? ov::Shape{outputs, channels} : ov::Shape{outputs, channels, 1, 1};
        const auto weight_type = compressed ? ov::element::i8 : ov::element::f16;
        std::vector<InputShape> shapes{input_shape};
        if (!constant_weights) {
            shapes.push_back({ov::PartialShape(weight_shape), std::vector<ov::Shape>(input_shape.second.size(), weight_shape)});
            weight_tensor = ov::Tensor(weight_type, weight_shape);
        }
        init_input_shapes(shapes);

        auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        ov::ParameterVector parameters{input};
        std::shared_ptr<ov::Node> weights;
        if (constant_weights) {
            weights = ov::opset13::Constant::create(weight_type, weight_shape, {1});
        } else {
            auto weight_input = std::make_shared<ov::opset13::Parameter>(weight_type, inputDynamicShapes[1]);
            parameters.push_back(weight_input);
            weights = weight_input;
        }
        if (compressed) {
            weights = std::make_shared<ov::opset13::Convert>(weights, ov::element::f16);
            const ov::Shape scale_shape = matmul ? ov::Shape{outputs, 1} : ov::Shape{outputs, 1, 1, 1};
            auto scale = ov::opset13::Constant::create(ov::element::f16, scale_shape, {0.25f});
            weights = std::make_shared<ov::opset13::Multiply>(weights, scale);
        }
        std::shared_ptr<ov::Node> result;
        if (matmul) {
            result = std::make_shared<ov::opset13::MatMul>(input, weights, false, true);
            auto order = ov::opset13::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
            result = std::make_shared<ov::opset13::Transpose>(result, order);
        } else {
            result = std::make_shared<ov::opset13::Convolution>(input,
                                                                weights,
                                                                ov::Strides{1, 1},
                                                                ov::CoordinateDiff{0, 0},
                                                                ov::CoordinateDiff{0, 0},
                                                                ov::Strides{1, 1});
        }
        function = std::make_shared<ov::Model>(ov::OutputVector{result}, parameters);
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_shapes) override {
        inputs.clear();
        const auto& parameters = function->get_parameters();
        ov::Tensor data(ov::element::f16, target_input_shapes[0]);
        std::fill_n(data.data<ov::float16>(), data.get_size(), ov::float16(1.0f));
        inputs[parameters[0]] = data;
        if (parameters.size() == 2) {
            // Keep the same tensor buffer and change only its contents between inferences.
            if (weight_tensor.get_element_type() == ov::element::i8) {
                std::fill_n(weight_tensor.data<int8_t>(), weight_tensor.get_size(), static_cast<int8_t>(iteration + 1));
            } else {
                std::fill_n(weight_tensor.data<ov::float16>(), weight_tensor.get_size(), ov::float16(static_cast<float>(iteration + 1)));
            }
            inputs[parameters[1]] = weight_tensor;
        }
        ++iteration;
    }

    void infer() override {
        // Reuse the request as well as the weight buffer to exercise cached weight handling.
        if (!request_initialized) {
            inferRequest = compiledModel.create_infer_request();
            request_initialized = true;
        }
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
    }

private:
    ov::Tensor weight_tensor;
    size_t iteration = 0;
    bool request_initialized = false;
};

}  // namespace ov::test::gpu
