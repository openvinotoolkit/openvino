// Copyright (C) 2018-2026 Intel Corporation
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

#include "openvino/op/mod.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"

#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"

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

    if (std::all_of(m_dilations.begin(), m_dilations.end(), [](size_t d) {
            return d == static_cast<size_t>(1);
        })) {
        return {std::make_shared<v1::AvgPool>(m_inputs.at(0),
                                              m_strides,
                                              m_padding_below,
                                              m_padding_above,
                                              m_kernel_shape,
                                              !count_include_pad,
                                              m_rounding_type,
                                              m_auto_pad)};
    }
    return {std::make_shared<v16::AvgPool>(m_inputs.at(0),
                                           m_strides,
                                           m_dilations,
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
        auto indices = max_pool->output(1);

        auto shape_of = std::make_shared<v3::ShapeOf>(m_inputs.at(0), element::i64);

        auto axes = v0::Constant::create(element::i64, Shape{}, {0});
        auto indices_h = v0::Constant::create(element::i64, Shape{1}, {2});
        auto indices_w = v0::Constant::create(element::i64, Shape{1}, {3});

        auto height_node = std::make_shared<v8::Gather>(shape_of, indices_h, axes);
        auto width_node  = std::make_shared<v8::Gather>(shape_of, indices_w, axes);

        auto col = std::make_shared<v1::Mod>(indices, width_node);

        auto row = std::make_shared<v1::Divide>(indices, width_node);

        auto col_times_height = std::make_shared<v1::Multiply>(col, height_node);
        auto new_indices = std::make_shared<v1::Add>(col_times_height, row);

        return {max_pool->output(0), new_indices};
    } else {
        return {max_pool->output(0), max_pool->output(1)};
    }
}
}  // namespace pooling
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
