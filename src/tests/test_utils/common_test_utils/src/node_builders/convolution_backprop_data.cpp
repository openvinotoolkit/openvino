// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution_backprop_data.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_convolution_backprop_data(const ov::Output<Node>& in,
                                                         const ov::element::Type& type,
                                                         const std::vector<size_t>& filter_size,
                                                         const std::vector<size_t>& strides,
                                                         const std::vector<ptrdiff_t>& pads_begin,
                                                         const std::vector<ptrdiff_t>& pads_end,
                                                         const std::vector<size_t>& dilations,
                                                         const ov::op::PadType& auto_pad,
                                                         size_t num_out_channels,
                                                         bool add_biases,
                                                         const std::vector<ptrdiff_t>& output_padding,
                                                         const std::vector<float>& filter_weights,
                                                         const std::vector<float>& biases_weights) {
    auto shape = in.get_partial_shape();
    ov::Shape filter_weights_shape = {static_cast<size_t>(shape[1].get_length()), num_out_channels};
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());

    std::shared_ptr<ov::op::v0::Constant> filter_weights_node;
    if (!filter_weights.empty()) {
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(type, filter_weights_shape, filter_weights);
    } else {
        auto tensor = create_and_fill_tensor(type, filter_weights_shape);
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    return make_convolution_backprop_data(in,
                                          filter_weights_node,
                                          type,
                                          strides,
                                          pads_begin,
                                          pads_end,
                                          dilations,
                                          auto_pad,
                                          add_biases,
                                          output_padding,
                                          biases_weights);
}

std::shared_ptr<ov::Node> make_convolution_backprop_data(const ov::Output<Node>& in,
                                                         const ov::Output<Node>& weights,
                                                         const ov::element::Type& type,
                                                         const std::vector<size_t>& strides,
                                                         const std::vector<ptrdiff_t>& pads_begin,
                                                         const std::vector<ptrdiff_t>& pads_end,
                                                         const std::vector<size_t>& dilations,
                                                         const ov::op::PadType& auto_pad,
                                                         bool add_biases,
                                                         const std::vector<ptrdiff_t>& output_padding,
                                                         const std::vector<float>& biases_weights) {
    auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(in,
                                                                        weights,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        dilations,
                                                                        auto_pad);

    if (!output_padding.empty()) {
        deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(in,
                                                                       weights,
                                                                       strides,
                                                                       pads_begin,
                                                                       pads_end,
                                                                       dilations,
                                                                       auto_pad,
                                                                       output_padding);
    }

    if (add_biases) {
        std::shared_ptr<ov::op::v0::Constant> biases_weights_node;
        if (!biases_weights.empty()) {
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, biases_weights);
        } else {
            auto tensor = create_and_fill_tensor(type, ov::Shape{});
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto add = std::make_shared<ov::op::v1::Add>(deconv, biases_weights_node);
        return add;
    } else {
        return deconv;
    }
}

std::shared_ptr<ov::Node> make_convolution_backprop_data(const ov::Output<Node>& in,
                                                         const ov::Output<Node>& outputShape,
                                                         const ov::element::Type& type,
                                                         const std::vector<size_t>& filter_size,
                                                         const std::vector<size_t>& strides,
                                                         const std::vector<ptrdiff_t>& pads_begin,
                                                         const std::vector<ptrdiff_t>& pads_end,
                                                         const std::vector<size_t>& dilations,
                                                         const ov::op::PadType& auto_pad,
                                                         size_t num_out_channels,
                                                         bool add_biases,
                                                         const std::vector<ptrdiff_t>& output_padding,
                                                         const std::vector<float>& filter_weights,
                                                         const std::vector<float>& biases_weights) {
    auto shape = in.get_partial_shape();
    ov::Shape filter_weights_shape = {static_cast<size_t>(shape[1].get_length()), num_out_channels};
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());

    std::shared_ptr<ov::op::v0::Constant> filter_weights_node;
    if (!filter_weights.empty()) {
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(type, filter_weights_shape, filter_weights);
    } else {
        auto tensor = create_and_fill_tensor(type, filter_weights_shape);
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(in,
                                                                        filter_weights_node,
                                                                        outputShape,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        dilations,
                                                                        auto_pad);

    if (!output_padding.empty()) {
        deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(in,
                                                                       filter_weights_node,
                                                                       outputShape,
                                                                       strides,
                                                                       pads_begin,
                                                                       pads_end,
                                                                       dilations,
                                                                       auto_pad,
                                                                       output_padding);
    }

    if (add_biases) {
        std::shared_ptr<ov::op::v0::Constant> biases_weights_node;
        if (!biases_weights.empty()) {
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, biases_weights);
        } else {
            auto tensor = create_and_fill_tensor(type, ov::Shape{});
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto add = std::make_shared<ov::op::v1::Add>(deconv, biases_weights_node);
        return add;
    } else {
        return deconv;
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
