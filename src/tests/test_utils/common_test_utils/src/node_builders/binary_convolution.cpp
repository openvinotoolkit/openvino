// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/binary_convolution.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/binary_convolution.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_binary_convolution(const ov::Output<Node>& in,
                                                  const std::vector<size_t>& filter_size,
                                                  const std::vector<size_t>& strides,
                                                  const std::vector<ptrdiff_t>& pads_begin,
                                                  const std::vector<ptrdiff_t>& pads_end,
                                                  const std::vector<size_t>& dilations,
                                                  const ov::op::PadType& auto_pad,
                                                  size_t num_out_channels,
                                                  float pad_value,
                                                  const std::vector<int8_t>& filter_weihgts) {
    auto shape = in.get_shape();
    ov::Shape filter_weights_shape = {num_out_channels, shape[1]};
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());

    auto filter_weights_node = std::make_shared<ov::op::v0::Constant>(element::u1, filter_weights_shape);
    const size_t byteNum = (ov::shape_size(filter_weights_shape) + 7) / 8;
    int8_t* buffer = const_cast<int8_t*>(filter_weights_node->get_data_ptr<int8_t>());
    if (filter_weihgts.size() == 0) {
        auto tensor = create_and_fill_tensor(ov::element::i8, filter_weights_shape);
        auto weights = static_cast<int8_t*>(tensor.data());
        for (size_t i = 0; i < byteNum; i++)
            buffer[i] = weights[i];
    } else {
        for (size_t i = 0; i < byteNum; i++)
            buffer[i] = filter_weihgts[i];
    }
    auto conv = std::make_shared<ov::op::v1::BinaryConvolution>(
        in,
        filter_weights_node,
        strides,
        pads_begin,
        pads_end,
        dilations,
        ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
        pad_value,
        auto_pad);
    return conv;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
