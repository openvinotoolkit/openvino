// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/pooling_factory.hpp"

#include <iterator>

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/transpose.hpp"
#include "utils/convpool.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace pooling {

namespace {
std::shared_ptr<v0::Constant> transposition_axis_order(const ov::Rank& input_rank) {
    FRONT_END_GENERAL_CHECK(input_rank.is_static(),
                            "Generating column-major MaxPool results is supported only for inputs with static rank.");

    const auto rank = static_cast<size_t>(input_rank.get_length());

    std::vector<int32_t> axes(rank);
    std::iota(axes.begin(), axes.end(), 0);
    std::reverse(axes.begin() + 2, axes.end());

    return std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{rank}, axes);
}
}  // namespace

PoolingFactory::PoolingFactory(const Node& node)
    : m_onnx_node{node},
      m_inputs{node.get_ov_inputs()},
      m_kernel_shape(node.get_attribute_value<std::vector<std::size_t>>("kernel_shape")),
      m_strides{convpool::get_strides(node, m_kernel_shape.size())},
      m_dilations{convpool::get_dilations(node, m_kernel_shape.size())},
      m_auto_pad{convpool::get_auto_pad(node)},
      m_rounding_type{convpool::get_rounding_type(node)} {
    const auto paddings = convpool::get_pads(node, m_kernel_shape.size());
    const ov::CoordinateDiff& padding_above{paddings.second};
    const ov::CoordinateDiff& padding_below{paddings.first};
    m_padding_below = ov::Shape{std::begin(padding_below), std::end(padding_below)};
    m_padding_above = ov::Shape{std::begin(padding_above), std::end(padding_above)};
    m_storage_order = static_cast<StorageOrder>(node.get_attribute_value<int64_t>("storage_order", 0));
}

ov::OutputVector PoolingFactory::make_avg_pool() const {
    const bool count_include_pad = m_onnx_node.get_attribute_value<std::int64_t>("count_include_pad", 0);
    return {std::make_shared<v1::AvgPool>(m_inputs.at(0),
                                          m_strides,
                                          m_padding_below,
                                          m_padding_above,
                                          m_kernel_shape,
                                          !count_include_pad,
                                          m_rounding_type,
                                          m_auto_pad)};
}

ov::OutputVector PoolingFactory::make_max_pool() const {
    return {std::make_shared<v1::MaxPool>(m_inputs.at(0),
                                          m_strides,
                                          m_padding_below,
                                          m_padding_above,
                                          m_kernel_shape,
                                          m_rounding_type,
                                          m_auto_pad)};
}

ov::OutputVector PoolingFactory::make_max_pool_with_indices() const {
    const auto max_pool = std::make_shared<v8::MaxPool>(m_inputs.at(0),
                                                        m_strides,
                                                        m_dilations,
                                                        m_padding_below,
                                                        m_padding_above,
                                                        m_kernel_shape,
                                                        m_rounding_type,
                                                        m_auto_pad);
    if (m_storage_order == StorageOrder::COLUMN_MAJOR) {
        const auto transposition_axes = transposition_axis_order(m_inputs.at(0).get_partial_shape().rank());
        const auto transposed_indices = std::make_shared<v1::Transpose>(max_pool->output(1), transposition_axes);

        return {max_pool->output(0), transposed_indices};
    } else {
        return {max_pool->output(0), max_pool->output(1)};
    }
}
}  // namespace pooling
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
