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

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
///   b = e^a/sum(N-1; j=0; e^j)
/// @par Where:
///   @li N : number of values to normalize
///   @li b : value after normalization
///   @li a : value before normalization
struct softmax : public primitive_base<softmax> {
    CLDNN_DECLARE_PRIMITIVE(softmax)

    /// @brief Enum type to specify softmax's normalization scope (see #dimension).
    enum dimension_t {
        normalize_b,
        normalize_f,
        normalize_x,
        normalize_y,
        normalize_z,
        normalize_fyx,
        normalize_all
    };

    /// @brief Constructs softmax primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param dimension Defines a scope of normalization (see #dimension).
    softmax(const primitive_id& id,
            const primitive_id& input,
            const dimension_t dimension = normalize_fyx,
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding), dimension(dimension) {}

    /// @brief Defines a scope of a single softmax normalization.
    /// @details
    /// Being given a 4-dimensional input, which consists of b,f,y,x dimensions, softmax normalizes data which are divided into multiple independent sets.
    /// Specific behaviour is determined by this parameter, as follows:
    /// - when set to @link softmax::dimension_t softmax::normalize_x @endlink each input row is normalized independently,
    /// - when set to @link softmax::dimension_t softmax::normalize_y @endlink each input column is normalized independently,
    /// - when set to @link softmax::dimension_t softmax::normalize_z @endlink each input z-coordinate is normalized independently,
    /// - when set to @link softmax::dimension_t softmax::normalize_f @endlink each in-depth vector of input is normalized independently,
    /// - when set to @link softmax::dimension_t softmax::normalize_fyx @endlink each 3d image within input is normalized independently,
    /// - when set to @link softmax::dimension_t softmax::normalize_all @endlink everything is normalized,
    dimension_t dimension;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
