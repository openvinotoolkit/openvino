// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/pooling_factory.hpp"

#include <iterator>

#include "exceptions.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
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

ov::Output<ov::Node> PoolingFactory::make_avg_pool_op(const ov::Output<ov::Node>& data, bool exclude_pad) const {
    const bool is_ceil_mode = m_rounding_type == ov::op::RoundingType::CEIL;

    const bool has_dilations = std::any_of(m_dilations.begin(), m_dilations.end(), [](size_t d) {
        return d != static_cast<size_t>(1);
    });

    if (!has_dilations && !is_ceil_mode) {
        return std::make_shared<v1::AvgPool>(data,
                                             m_strides,
                                             m_padding_below,
                                             m_padding_above,
                                             m_kernel_shape,
                                             exclude_pad,
                                             m_rounding_type,
                                             m_auto_pad);
    }
    // ONNX ceil_mode must not let a window start in padding; only CEIL_TORCH enforces that.
    const auto rounding_type = is_ceil_mode ? ov::op::RoundingType::CEIL_TORCH : m_rounding_type;
    return std::make_shared<v16::AvgPool>(data,
                                          m_strides,
                                          m_dilations,
                                          m_padding_below,
                                          m_padding_above,
                                          m_kernel_shape,
                                          exclude_pad,
                                          rounding_type,
                                          m_auto_pad);
}

ov::OutputVector PoolingFactory::make_avg_pool() const {
    const bool count_include_pad = m_onnx_node.get_attribute_value<std::int64_t>("count_include_pad", 0);
    return {make_avg_pool_op(m_inputs.at(0), !count_include_pad)};
}

ov::OutputVector PoolingFactory::make_lp_pool(float p_norm) const {
    CHECK_VALID_NODE(m_onnx_node, p_norm > 0.f, "Only positive values are supported for 'p' attribute.");

    const auto& data = m_inputs.at(0);

    // Lp pooling is a sum of |x|^p over a pooling window, followed by the p-th root.
    // The sum is obtained from an average pooling which always divides by the (constant)
    // kernel volume, hence exclude_pad has to be disabled.
    // The norm is computed in f32 regardless of the input's element type to avoid overflow
    // or precision loss for reduced-precision inputs (e.g. fp16 can already overflow with
    // p=2 once |x| > 256), then converted back to the original type at the end.
    // f64 is deliberately not preserved here: Abs/Power reference evaluate() only support
    // {f32, i32, i64, u32, u64}, so an f64 Abs/Power would not be executable by any backend
    // (including the interpreter/template one), unlike a plain f64 AveragePool.
    ov::Output<ov::Node> pooled = std::make_shared<v0::Convert>(data, ov::element::f32);
    pooled = std::make_shared<v0::Abs>(pooled);
    if (p_norm != 1.f) {
        const auto p_const = v0::Constant::create(ov::element::f32, ov::Shape{}, {p_norm});
        pooled = std::make_shared<v1::Power>(pooled, p_const);
    }

    pooled = make_avg_pool_op(pooled, false);

    const auto kernel_volume =
        v0::Constant::create(ov::element::f32, ov::Shape{}, {static_cast<float>(shape_size(m_kernel_shape))});
    pooled = std::make_shared<v1::Multiply>(pooled, kernel_volume);

    if (p_norm != 1.f) {
        const auto inv_p = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.f / p_norm});
        pooled = std::make_shared<v1::Power>(pooled, inv_p);
    }

    return {std::make_shared<v1::ConvertLike>(pooled, data)};
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
