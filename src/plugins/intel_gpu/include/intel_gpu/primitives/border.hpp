// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include "openvino/core/coordinate_diff.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Adds border around input.
///
/// @details Applies border of specified type around input data. The size of output data is increased
///          by @c pads_begin and by @c pads_end.
/// @n
/// @n@b Requirements:
/// @n - @c pads_begin and @c pads_end must be non-negative on all dimensions and compatible
///      with size of input (describe the same dimensions).
/// @n - For @c PadMode equal to @c SYMMETRIC, @c pads_begin and @c pads_end
///      must be lower than or equal to size of input on corresponding dimension (for all dimensions)
/// @n - For @c PadMode equal to @c REFLECT, @c pads_begin and @c pads_end
///      must be lower than size of input on corresponding dimension (for all dimensions)
/// @n Breaking any of this conditions will cause exeption throw.
struct border : public primitive_base<border> {
    CLDNN_DECLARE_PRIMITIVE(border)

    /// @brief Constructs border primitive / layer with static pads.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param input              An identifier of primitive which is an input for newly created
    ///                           border primitive.
    /// @param pads_begin         Sizes of border that needs to be added from left
    ///                           (in X dimension) and from top (in Y dimension).
    /// @param pads_end           Sizes of border that needs to be added from right
    ///                           (in X dimension) and from bottom (in Y dimension).
    /// @param type               Type of added border.
    /// @param pad_mode           Value of elements which is used for paddings
    /// @param output_padding     Optional padding for output from primitive.
    border(const primitive_id& id,
           const input_info& input,
           const ov::CoordinateDiff& pads_begin = {0, 0, 0, 0},
           const ov::CoordinateDiff& pads_end = {0, 0, 0, 0},
           const ov::op::PadMode pad_mode = ov::op::PadMode::CONSTANT,
           const float pad_value = 0.0f,
           const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          pads_begin(pads_begin),
          pads_end(pads_end),
          pad_mode(pad_mode),
          pad_value(pad_value) {}

    /// @brief Constructs border primitive / layer with dynamic pads.
    border(const primitive_id& id,
           const input_info& input,
           const input_info& pads_begin_id,
           const input_info& pads_end_id,
           const ov::op::PadMode pad_mode = ov::op::PadMode::CONSTANT,
           const float pad_value = 0.0f,
           const padding& output_padding = padding())
        : primitive_base(id, {input, pads_begin_id, pads_end_id}, {output_padding}),
          pads_begin({}),
          pads_end({}),
          pad_mode(pad_mode),
          pad_value(pad_value) {}

    /// @brief Sizes of border that needs to be added from left (in X dimension) and from top (in Y dimension).
    ov::CoordinateDiff pads_begin;
    /// @brief Sizes of border that needs to be added from right (in X dimension) and from bottom (in Y dimension).
    ov::CoordinateDiff pads_end;
    /// @brief Type of border that needs to be added to the input.
    ov::op::PadMode pad_mode;
    /// @brief Border value that is used in constant mode.
    float pad_value;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
