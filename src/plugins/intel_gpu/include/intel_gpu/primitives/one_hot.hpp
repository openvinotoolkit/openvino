// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Creates a one-hot encoding of the input.
/// @details Creates a one-hot encoding of the input, putting the new one-hot axis in the position
/// @n       specified by the @p one_hot_axis input, using the @p shape tensor as size reference.
/// @n       The size of @p shape must be appropriate for adding a one-hot axis to input. For example,
/// @n      <tt>input_sizes = (1, in_f, in_y, in_x)</tt>
/// @n expanded with
/// @n      <tt>one_hot_axis = 2</tt>
/// @n would insert the one-hot axis in the Y dimension, requiring
/// @n      <tt>shape = (in_f, in_y, one-hot_limit, in_x)</tt>
/// @n The output values would then be determined by input as
/// @n      <tt>output[f, y, i, x] = (input[0, f, y, x] == i) ? 1 : 0;</tt>
/// @n Since determining whether the input is appropriate (that the one-hot axis
/// @n has enough space to fully encode all inputs) requires scanning the whole
/// @n input, the primitive doesn't check for that, instead producing all-zeros
/// @n output axes for inputs below 0 and greater than the limit set by
/// @n @p shape.
/// @n
/// @n@b Requirements
/// @n - @p one_hot_axis must be within (inclusive) range 0 - 3.
/// @n - @p shape must fit input sizes (see example above).
/// @n - input batch size must be equal to 1.
/// @n
/// @n Breaking any of this conditions will cause exception throw.
struct one_hot : public primitive_base<one_hot> {
    CLDNN_DECLARE_PRIMITIVE(one_hot)

    /// @brief Constructs one-hot primitive layer.
    /// @param id              An identifier of new primitive.
    /// @param input           An identifier of primitive which is an input for newly created one-hot primitive.
    /// @param shape           Size of the output primitive.
    /// @param one_hot_axis    One-hot axis position (0-based, from left to right) in shape.
    /// @param output_padding  Optional padding for output from primitive.
    one_hot(const primitive_id& id,
            const primitive_id& input,
            const tensor& shape,
            const int64_t& one_hot_axis,
            const int64_t& depth,
            const float& on_value = 1.0f,
            const float& off_value = 0.0f,
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding)
        , shape(shape)
        , one_hot_axis(one_hot_axis)
        , depth(depth)
        , on_value(on_value)
        , off_value(off_value) {}

    /// @brief Constructs one-hot primitive layer.
    /// @param id              An identifier of new primitive.
    /// @param input           An identifier of primitive which is an input for newly created one-hot primitive.
    /// @param shape           Size of the output primitive.
    /// @param output_dt       Data type of output elements.
    /// @param one_hot_axis    One-hot axis position (0-based, from left to right) in shape.
    /// @param output_padding  Optional padding for output from primitive.
    one_hot(const primitive_id& id,
            const primitive_id& input,
            const tensor& shape,
            const data_types output_dt,
            const int64_t& one_hot_axis,
            const int64_t& depth,
            const float& on_value = 1.0f,
            const float& off_value = 0.0f,
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding, optional_data_type{output_dt})
        , shape(shape)
        , one_hot_axis(one_hot_axis)
        , depth(depth)
        , on_value(on_value)
        , off_value(off_value) {}

    /// @brief Output size reference.
    tensor shape;
    /// @brief One-hot axis position in output shape (0-based, from left to right).
    int64_t one_hot_axis;
    /// @brief The number of classes and thus the size of the one-hot dimension
    int64_t depth;
    /// @brief The locations represented by indices in indices take this value.
    float on_value;
    /// @brief all other locations take value this value.
    float off_value;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
