// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_convolution(const ov::Output<Node>& in,
                                           const ov::element::Type& type,
                                           const std::vector<size_t>& filter_size,
                                           const std::vector<size_t>& strides,
                                           const std::vector<ptrdiff_t>& pads_begin,
                                           const std::vector<ptrdiff_t>& pads_end,
                                           const std::vector<size_t>& dilations,
                                           const ov::op::PadType& auto_pad,
                                           size_t num_out_channels,
                                           bool add_biases,
                                           const std::vector<float>& filter_weights,
                                           const std::vector<float>& biases_weights) {
    auto shape = in.get_partial_shape();
    ov::Shape filter_weights_shape = {num_out_channels, static_cast<size_t>(shape[1].get_length())};
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());

    std::shared_ptr<ov::op::v0::Constant> filter_weights_node;
    if (!filter_weights.empty()) {
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(type, filter_weights_shape, filter_weights);
    } else {
        auto tensor = create_and_fill_tensor(type, filter_weights_shape, 9, 1);
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    auto conv = std::make_shared<ov::op::v1::Convolution>(in,
                                                          filter_weights_node,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          dilations,
                                                          auto_pad);
    if (add_biases) {
        std::shared_ptr<ov::op::v0::Constant> biases_weights_node;
        const size_t rank = in.get_partial_shape().rank().get_length();
        ov::Shape bias_shape(rank, 1);
        bias_shape[1] = num_out_channels;
        if (!biases_weights.empty()) {
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(type, bias_shape, biases_weights);
        } else {
            auto tensor = create_and_fill_tensor(type, bias_shape, 9, 1);
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto add = std::make_shared<ov::op::v1::Add>(conv, biases_weights_node);
        return add;
    } else {
        return conv;
    }
}

std::shared_ptr<ov::Node> make_convolution(const ov::Output<Node>& in_data,
                                           const ov::Output<Node>& in_weights,
                                           const ov::element::Type& type,
                                           const std::vector<size_t>& filter_size,
                                           const std::vector<size_t>& strides,
                                           const std::vector<ptrdiff_t>& pads_begin,
                                           const std::vector<ptrdiff_t>& pads_end,
                                           const std::vector<size_t>& dilations,
                                           const ov::op::PadType& auto_pad,
                                           size_t num_out_channels,
                                           bool add_biases,
                                           const std::vector<float>& biases_weights) {
    auto shape = in_data.get_partial_shape();
    auto conv = std::make_shared<ov::op::v1::Convolution>(in_data,
                                                          in_weights,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          dilations,
                                                          auto_pad);
    if (add_biases) {
        std::shared_ptr<ov::op::v0::Constant> biases_weights_node;
        const size_t rank = in_data.get_partial_shape().rank().get_length();
        ov::Shape bias_shape(rank, 1);
        bias_shape[1] = num_out_channels;
        if (!biases_weights.empty()) {
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(type, bias_shape, biases_weights);
        } else {
            auto tensor = create_and_fill_tensor(type, bias_shape, 9, 1);
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto add = std::make_shared<ov::op::v1::Add>(conv, biases_weights_node);
        return add;
    } else {
        return conv;
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
