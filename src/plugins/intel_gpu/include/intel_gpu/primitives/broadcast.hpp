// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/partial_shape.hpp"
#include "openvino/op/broadcast.hpp"

#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Broadcasts input to defined by @p broadcast_sizes output. @p broadcast_axes are used to
///        reinterpret input (reshape) inside algorithm.
///
/// @details Takes input, reinterpret it according to @p broadcast_axes
///          and copies it to output once or multiple times.
/// @n
/// @n Simple example with empty @p broadcast_axes. Lets assume that:
/// @n      <tt>input_sizes = (in_b, in_f, in_y, in_x)</tt>
/// @n      <tt>broadcast_sizes = (bs_b, bs_f, bs_y, bs_x)</tt>
/// @n      <tt>broadcast_axes = () - empty</tt>
/// @n The input is broadcasted on each dimension where <tt>bs_{dim} > in_{dim}</tt> and <tt>bs_{dim}</tt>
///    is dividable by <tt>in_{dim}</tt> (input is copied <tt>bs_{dim} / in_{dim}</tt> times).
///    The dimensions where <tt>bs_{dim}</tt> is equal to <tt>in_{dim}</tt> remain unchanged.
/// @n The resulting output will have sizes equal to @p broadcast_sizes and contains values from
///    input that meet following criteria:
/// @n      <tt>output[(b, f, y, x)] = input[(b % in_b, f % in_f, y % in_y, x % in_x)]</tt>
/// @n where <tt>(b, f, y, x)</tt> is a position of value in a primitive output.
/// @n
/// @n More complicated example with non empty @p broadcast_axes. Lets assume that:
/// @n      <tt>broadcast_sizes = (bs_b, bs_f, bs_y, bs_x)</tt>
/// @n      <tt>broadcast_axes = (2)</tt>
/// @n Taking into account broadcast_axes size (=1) primitive's input must be (4 - 1 = 3):
/// @n      <tt>primitive input = (1, in_b, in_f, in_x)</tt>
/// @n Due to broadcast_axes = (2) primitive will interpret input as:
/// @n      <tt>primitive input(internal representation) = (in_b, in_f, 1, in_x)</tt>
/// @n Now, you can apply broadcast rules from previous example to modified (reinterpreted)
///    input and output:
/// @n      <tt>input_sizes = (in_b, in_f, 1, in_x)</tt>
/// @n      <tt>output_shape = (bs_b, bs_f, bs_y, bs_x)</tt>
/// @n      <tt>broadcast_axes = () - empty</tt>
/// @n
/// @n@b Requirements:
/// @n - @p broadcast_sizes must be positive on all dimensions.
/// @n - @p broadcast_axes size (dimensions count) must be within (inclusive) range
///      0 - 4.
/// @n - @p broadcast_axes mustn't have duplicate values.
/// @n - Values of @p broadcast_axes must be within (inclusive) range 0 - 3
/// @n - @p output_shape must be greater (dividable) than or equal to reinterpreted
///      input on all dimensions.
/// @n Breaking any of these conditions will raise an exception.
struct broadcast : public primitive_base<broadcast> {
    CLDNN_DECLARE_PRIMITIVE(broadcast)

    broadcast() : primitive_base("", {}) {}

    /// @brief Constructs broadcast primitive / layer.
    ///
    /// @param id              An identifier of new primitive.
    /// @param input           An identifier of primitive which is an input for newly created
    ///                        broadcast primitive.
    /// @param broadcast_sizes Sizes of broadcast. Output size of current primitive
    ///                        will match broadcast sizes (layout type will not change).
    /// @param broadcast_axes  Axes positions (0-based, from left to right) in output_shape
    ///                        that are being broadcast. Values of broadcast_axes on remaining
    ///                        axes must be greater (dividable) or equal to corresponding input
    ///                        dimension values.
    broadcast(const primitive_id& id,
              const input_info& input,
              const tensor& broadcast_sizes,
              const std::vector<uint16_t>& broadcast_axes = {})
        : primitive_base(id, {input}),
          broadcast_sizes(broadcast_sizes),
          broadcast_axes(broadcast_axes) {}

    /// @brief Constructs broadcast primitive / layer with static target_shape.
    ///
    /// @param id             An identifier of new primitive.
    /// @param input          An identifier of primitive which is an input for newly created
    ///                       broadcast primitive.
    /// @param target_shape   The shape of the output tensor.
    /// @param axes_mapping   The axis positions (0-based) in the result that correspond
    ///                       to input axes. 'Arg' tensor is broadcast along the
    ///                       remaining axes.
    ///                       E.g., Input Shape - [3, 4], Target Shape - [3, 5, 4, 4]
    ///                       axes_mapping - [0, 2] => Broadcast along axes 1 and 3.
    ///                       axes_mapping - [0, 3] => Broadcast along axes 1 and 2.
    /// @param broadcast_spec Broadcast specification to use for determining broadcast
    ///                       axes. 'axes_mapping' should not be provided if mode other
    ///                       than explicit (none) is used.
    broadcast(const primitive_id& id,
              const input_info& input,
              const ov::Shape& target_shape,
              const ov::AxisSet& axes_mapping,
              const ov::op::BroadcastModeSpec& broadcast_spec = ov::op::BroadcastType::EXPLICIT)
        : primitive_base(id, {input}),
          target_shape(target_shape),
          axes_mapping(axes_mapping),
          broadcast_mode(broadcast_spec),
          broadcast_sizes(target_shape.empty() ? tensor(1) : tensor(0)),
          broadcast_axes({}) {}

    /// @brief Constructs broadcast primitive / layer with dynamic target_shape.
    broadcast(const primitive_id& id,
          const input_info& input,
          const input_info& target_shape_id,
          const ov::AxisSet& axes_mapping,
          const ov::op::BroadcastModeSpec& broadcast_spec = ov::op::BroadcastType::EXPLICIT)
    : primitive_base(id, {input, target_shape_id}),
      target_shape({}),
      axes_mapping(axes_mapping),
      broadcast_mode(broadcast_spec),
      broadcast_sizes({}),
      broadcast_axes({}) {}

    /// @brief The shape of the output tensor.
    ov::Shape target_shape;
    /// @brief The axis positions (0-based) in the result that correspond to input axes.
    ov::AxisSet axes_mapping;
    /// @brief Broadcast mode to use for determining broadcast axes.
    ov::op::BroadcastModeSpec broadcast_mode;
    /// @brief Expected sizes of output from broadcast primitive.
    tensor broadcast_sizes;
    /// @brief Array of axes positions from output shape (0-based, from left to right)
    ///        along which broadcast should happen.
    std::vector<uint16_t> broadcast_axes;

    ov::PartialShape output_pshape = ov::PartialShape::dynamic();

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, broadcast_axes.begin(), broadcast_axes.end());
        seed = hash_range(seed, axes_mapping.begin(), axes_mapping.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const broadcast>(rhs);

        return axes_mapping == rhs_casted.axes_mapping &&
               broadcast_mode == rhs_casted.broadcast_mode &&
               broadcast_sizes == rhs_casted.broadcast_sizes &&
               output_pshape == rhs_casted.output_pshape;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<broadcast>::save(ob);
        ob << target_shape;
        ob << axes_mapping;
        ob << make_data(&broadcast_mode, sizeof(ov::op::BroadcastModeSpec));
        ob << broadcast_sizes;
        ob << broadcast_axes;
        ob << output_pshape;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<broadcast>::load(ib);
        ib >> target_shape;
        ib >> axes_mapping;
        ib >> make_data(&broadcast_mode, sizeof(ov::op::BroadcastModeSpec));
        ib >> broadcast_sizes;
        ib >> broadcast_axes;
        ib >> output_pshape;
    }
};
}  // namespace cldnn
