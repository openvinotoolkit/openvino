// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/group_convolution.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_group_convolution(const ov::Output<Node>& in,
                                                 const ov::element::Type& type,
                                                 const std::vector<size_t>& filter_size,
                                                 const std::vector<size_t>& strides,
                                                 const std::vector<ptrdiff_t>& pads_begin,
                                                 const std::vector<ptrdiff_t>& pads_end,
                                                 const std::vector<size_t>& dilations,
                                                 const ov::op::PadType& auto_pad,
                                                 size_t num_out_channels,
                                                 size_t num_groups,
                                                 bool add_biases,
                                                 const std::vector<float>& filter_weights,
                                                 const std::vector<float>& biases_weights) {
    auto shape = in.get_partial_shape();
    ov::Shape filter_weights_shape = {num_out_channels, static_cast<size_t>(shape[1].get_length())};
    OPENVINO_ASSERT(!(filter_weights_shape[0] % num_groups || filter_weights_shape[1] % num_groups),
                    "incorrected shape for GroupConvolution");
    filter_weights_shape[0] /= num_groups;
    filter_weights_shape[1] /= num_groups;
    filter_weights_shape.insert(filter_weights_shape.begin(), num_groups);
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());

    std::shared_ptr<ov::op::v0::Constant> filter_weights_node;
    if (!filter_weights.empty()) {
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(type, filter_weights_shape, filter_weights);
    } else {
        auto tensor = create_and_fill_tensor(type, filter_weights_shape);
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    return make_group_convolution(in,
                                  filter_weights_node,
                                  type,
                                  strides,
                                  pads_begin,
                                  pads_end,
                                  dilations,
                                  auto_pad,
                                  add_biases,
                                  biases_weights);
}

std::shared_ptr<ov::Node> make_group_convolution(const ov::Output<Node>& in,
                                                 const ov::Output<Node>& weights,
                                                 const ov::element::Type& type,
                                                 const std::vector<size_t>& strides,
                                                 const std::vector<ptrdiff_t>& pads_begin,
                                                 const std::vector<ptrdiff_t>& pads_end,
                                                 const std::vector<size_t>& dilations,
                                                 const ov::op::PadType& auto_pad,
                                                 bool add_biases,
                                                 const std::vector<float>& biases_weights) {
    auto conv =
        std::make_shared<ov::op::v1::GroupConvolution>(in, weights, strides, pads_begin, pads_end, dilations, auto_pad);
    if (add_biases) {
        std::shared_ptr<ov::op::v0::Constant> biases_weights_node;
        if (!biases_weights.empty()) {
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, biases_weights);
        } else {
            auto tensor = create_and_fill_tensor(type, ov::Shape{});
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
