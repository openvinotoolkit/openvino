// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/multinomial.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/op/multinomial.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector multinomial(const ov::frontend::onnx::Node& node) {
    const auto input = node.get_ov_inputs().at(0);
    int sample_size = node.get_attribute_value<int>("sample_size", 1);

    auto input_shape = input.get_partial_shape();
    int batch_size = input_shape[0].get_length();
    int class_size = input_shape[1].get_length();

    std::vector<std::vector<int>> samples(batch_size, std::vector<int>(sample_size, 0));

    for (int i = 0; i < batch_size; ++i) {
        std::vector<double> probabilities(class_size);
        auto data_ptr = input.get_tensor().data<double>();
        for (int j = 0; j < class_size; ++j) {
            probabilities[j] = static_cast<double>(data_ptr[j]);
        }
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

        for (int j = 0; j < sample_size; ++j) {
            std::random_device rd;
            std::mt19937 gen(rd());
            samples[i][j] = distribution(gen);
        }
    }

    ov::OutputVector output;
    for (const auto& sample : samples) {
        auto constant_tensor =
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, static_cast<size_t>(sample_size)}, sample);
        auto output_tensor = std::make_shared<ov::op::v13::Multinomial>(input,
                                                                        constant_tensor,
                                                                        ov::element::Type_t::undefined,
                                                                        false,
                                                                        false,
                                                                        0,
                                                                        0);
        output.push_back(output_tensor);
    }

    return output;
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
