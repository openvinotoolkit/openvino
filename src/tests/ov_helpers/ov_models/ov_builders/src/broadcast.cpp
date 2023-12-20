// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/ov_builders/broadcast.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "ov_models/ov_builders/reshape.hpp"

namespace ov {
namespace op {
namespace util {
namespace {
///
/// \brief      Reconstructs axes mapping vector for Broadcast:v1 operation.
///
/// \param[in]  output_shape    The output shape of Broadcast operation.
/// \param[in]  broadcast_axes  The broadcast axes used for Broadcast:v0 operator.
///
/// \return     The vector with axes indexes mapping .
///
std::vector<size_t> get_axes_mapping(const Shape& output_shape, const AxisSet& broadcast_axes) {
    NGRAPH_CHECK((broadcast_axes.size() <= output_shape.size()));
    std::vector<size_t> axes_mapping(output_shape.size());
    iota(axes_mapping.begin(), axes_mapping.end(), 0);
    for (auto i = broadcast_axes.rbegin(); i != broadcast_axes.rend(); ++i) {
        axes_mapping.erase(axes_mapping.begin() + *i);
    }
    return axes_mapping;
}

///
/// \brief      Creates Node returning the axes mapping for Broadcast:v1 operation.
///
/// \param[in]  output_shape    The output shape of Broadcast operation.
/// \param[in]  broadcast_axes  The broadcast axes used for Broadcast:v0 operator.
///
/// \return     The Output object with Node returning axes mapping.
///
Output<Node> get_axes_mapping_output(const Shape& output_shape, const AxisSet& broadcast_axes) {
    std::vector<size_t> axes_mapping{get_axes_mapping(output_shape, broadcast_axes)};
    return ov::op::v0::Constant::create(element::i64, Shape{axes_mapping.size()}, axes_mapping);
}

static Output<Node> get_axes_mapping_output(const PartialShape& output_shape,
                                            const Output<Node>& input_shape,
                                            std::size_t start_match_axis) {
    const auto one_node = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    const auto zero_node = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto start_match_axis_node = ov::op::v0::Constant::create(element::i64, Shape{}, {start_match_axis});
    const auto target_shape_rank_node =
        ov::op::util::reshape(std::make_shared<ov::op::v3::ShapeOf>(input_shape), Shape{});

    const auto range_node =
        std::make_shared<ov::op::v4::Range>(zero_node, target_shape_rank_node, one_node, element::i64);

    // workaround for GPU plugin type incompatibility
    const auto range_node_converted =
        std::make_shared<ov::op::v0::Convert>(range_node, start_match_axis_node->get_element_type());
    // end of workaround

    const auto result = std::make_shared<ov::op::v1::Add>(range_node_converted, start_match_axis_node);
    return result;
}
}  // namespace

Output<Node> make_broadcast(const Output<Node>& node, const Shape& target_shape, const AxisSet& broadcast_axes) {
    return std::make_shared<ov::op::v1::Broadcast>(
        node,
        ov::op::v0::Constant::create(element::i64, Shape{target_shape.size()}, target_shape),
        get_axes_mapping_output(target_shape, broadcast_axes));
}

Output<Node> make_broadcast(const Output<Node>& node, const Shape& target_shape, size_t start_match_axis) {
    const auto node_shape = std::make_shared<ov::op::v3::ShapeOf>(node);
    return std::make_shared<ov::op::v1::Broadcast>(
        node,
        ov::op::v0::Constant::create(element::i64, Shape{target_shape.size()}, target_shape),
        get_axes_mapping_output(target_shape, node_shape, start_match_axis));
}
}  // namespace util
}  // namespace op
}  // namespace ov
