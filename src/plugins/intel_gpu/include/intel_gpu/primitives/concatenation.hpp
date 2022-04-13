// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @details Concatenation is used to concatenate multiple sources into one destination along specified dimension.
/// @notes
/// - all other dimensions (except the one along which concatenation take place) must have the same value in each source.
/// - order of arguments in primitive creation has impact on order of feature maps in output primitive.
///
/// @par Alogrithm:
/// \code
///     int outputIdx = 0
///     for(i : input)
///     {
///         for(f : i.features)
///         {
///             output[outputIdx] = f
///             outputIdx += 1
///         }
///     }
/// \endcode
/// @par Where:
///   @li input : data structure holding all source inputs for this primitive
///   @li output : data structure holding output data for this primitive
///   @li i.features : number of features in currently processed input
///   @li outputIdx : index of destination feature
struct concatenation : public primitive_base<concatenation> {
    CLDNN_DECLARE_PRIMITIVE(concatenation)

    enum concatenation_axis {
        along_b,
        along_f,
        along_x,
        along_y,
        along_z,
        along_w
    };

    /// @li Constructs concatenation primitive.
    /// @param id This primitive id.
    /// @param input Vector of input primitives ids.
    /// @param axis Selected dimension for concatenation.
    concatenation(
        const primitive_id& id,
        const std::vector<input_info>& input,
        const concatenation_axis axis,
        const primitive_id& ext_prim_id = "",
        const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, {output_padding}), axis(axis) {}

    /// @li Constructs concatenation primitive.
    /// @param id This primitive id.
    /// @param input Vector of input primitives ids.
    /// @param axis Selected dimension for concatenation.
    /// @param output_dt Data type of output tensor
    concatenation(
        const primitive_id& id,
        const std::vector<input_info>& input,
        const concatenation_axis axis,
        const data_types output_dt,
        const primitive_id& ext_prim_id = "",
        const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, {output_padding}, {optional_data_type{output_dt}}), axis(axis) {}

    /// @brief Dimension along which concatenation should take place
    concatenation_axis axis;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
