// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"

#include <cstddef>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"

namespace ov {
namespace test {
namespace utils {

static constexpr size_t channel_id = 1;

static ov::Shape get_filter_weights_shape(const ov::PartialShape& in_shape,
                                          const std::vector<size_t>& filter_size,
                                          size_t num_out_channels) {
    ov::Shape filter_weights_shape = {num_out_channels, static_cast<size_t>(in_shape[channel_id].get_length())};
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());
    return filter_weights_shape;
}

static std::shared_ptr<ov::Node> create_bias_op(const std::shared_ptr<ov::Node>& in,
                                                const ov::element::Type& type,
                                                size_t num_out_channels,
                                                const std::vector<float>& biases_weights) {
    std::shared_ptr<ov::op::v0::Constant> biases_weights_node;
    const auto rank = in->get_output_partial_shape(0).size();
    ov::Shape bias_shape(rank, 1);
    bias_shape[channel_id] = num_out_channels;
    if (!biases_weights.empty()) {
        biases_weights_node = std::make_shared<ov::op::v0::Constant>(type, bias_shape, biases_weights);
    } else {
        auto tensor = create_and_fill_tensor(type, bias_shape, 9, 1);
        biases_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    return std::make_shared<ov::op::v1::Add>(in, biases_weights_node);
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
    auto conv = std::make_shared<ov::op::v1::Convolution>(in_data,
                                                          in_weights,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          dilations,
                                                          auto_pad);

    if (add_biases) {
        return create_bias_op(conv, type, num_out_channels, biases_weights);
    }

    return conv;
}

std::shared_ptr<ov::Node> make_convolution(const ov::Output<Node>& in,
                                           const ov::element::Type& type,
                                           const std::vector<size_t>& filter_size,
                                           const std::vector<size_t>& strides,
                                           const std::vector<ptrdiff_t>& pads_begin,
                                           const std::vector<ptrdiff_t>& pads_end,
                                           const std::vector<size_t>& dilations,
                                           const ov::op::PadType& auto_pad,
                                           size_t num_out_channels,
                                           const std::optional<InputGenerateData>& input_data,
                                           bool add_biases,
                                           const std::vector<float>& biases_weights) {
    auto filter_weights_shape = get_filter_weights_shape(in.get_partial_shape(), filter_size, num_out_channels);

    static const auto defaultWeightsData = ov::test::utils::InputGenerateData{1, 9};
    auto tensor = create_and_fill_tensor(type, filter_weights_shape, input_data.value_or(defaultWeightsData));
    auto filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);

    return make_convolution(in,
                            filter_weights_node,
                            type,
                            filter_size,
                            strides,
                            pads_begin,
                            pads_end,
                            dilations,
                            auto_pad,
                            num_out_channels,
                            add_biases,
                            biases_weights);
}

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
    if (filter_weights.empty()) {  // if no weights data provided, generate default weights
        return make_convolution(in,
                                type,
                                filter_size,
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                num_out_channels,
                                {},
                                add_biases,
                                biases_weights);
    }

    ov::Shape filter_weights_shape = get_filter_weights_shape(in.get_partial_shape(), filter_size, num_out_channels);
    auto filter_weights_node = std::make_shared<ov::op::v0::Constant>(type, filter_weights_shape, filter_weights);

    return make_convolution(in,
                            filter_weights_node,
                            type,
                            filter_size,
                            strides,
                            pads_begin,
                            pads_end,
                            dilations,
                            auto_pad,
                            num_out_channels,
                            add_biases,
                            biases_weights);
}
}  // namespace utils
}  // namespace test
}  // namespace ov
