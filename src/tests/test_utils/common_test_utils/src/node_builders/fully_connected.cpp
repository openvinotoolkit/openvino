// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/fully_connected.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_fully_connected(const ov::Output<Node>& in,
                                               const ov::element::Type& type,
                                               const size_t output_size,
                                               bool addBias,
                                               const ov::Shape& weights_shape,
                                               const std::vector<float>& weights,
                                               const std::vector<float>& bias_weights) {
    auto shape = weights_shape;
    if (shape.empty()) {
        auto input_shape = in.get_shape();
        shape = {input_shape[1], output_size};
    }

    std::shared_ptr<ov::op::v0::Constant> weights_node;
    if (weights.empty()) {
        auto tensor = ov::test::utils::create_and_fill_tensor(type, shape);
        weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
    } else {
        weights_node = std::make_shared<ov::op::v0::Constant>(type, shape, weights);
    }

    auto fc = std::make_shared<ov::op::v0::MatMul>(in, weights_node, false, false);
    fc->set_friendly_name("FullyConnected");

    if (addBias) {
        std::shared_ptr<ov::op::v0::Constant> bias_weights_node;
        if (bias_weights.empty()) {
            auto tensor = ov::test::utils::create_and_fill_tensor(type, ov::Shape{});
            bias_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
        } else {
            bias_weights_node = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, bias_weights);
        }

        auto add = std::make_shared<ov::op::v1::Add>(fc, bias_weights_node);

        return add;
    } else {
        return fc;
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
