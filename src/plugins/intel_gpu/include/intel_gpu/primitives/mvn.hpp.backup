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

/// @brief Mean Variance Normalization primitive.
/// @details Normalizes the input to have 0-mean and/or unit (1) variance.
struct mvn : public primitive_base<mvn> {
    CLDNN_DECLARE_PRIMITIVE(mvn)

    /// @brief Constructs mvn primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param across_channels Determines if the normalization is done across or within channels. Default is within channels.'
    /// @param normalize_variance Determines if normalize variance is applied. Default is true.
    /// @param epsilon Epsilon for not dividing by zero while normalizing.
    /// @param eps_inside_sqrt The mode of applying epsilon.
    mvn(const primitive_id& id,
        const primitive_id& input,
        const bool normalize_variance,
        const float epsilon,
        const bool eps_inside_sqrt,
        const bool across_channels = false,
        const primitive_id& ext_prim_id = "",
        const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          normalize_variance(normalize_variance),
          epsilon(epsilon),
          eps_inside_sqrt(eps_inside_sqrt),
          across_channels(across_channels) {}

    /// @brief Determines if normalize variance is applied.
    bool normalize_variance;
    /// @brief Epsilon for not dividing by zero while normalizing.
    float epsilon;
    /// @brief The mode of applying epsilon.
    bool eps_inside_sqrt;
    /// @brief Determines if the normalization is done across or within channels.
    bool across_channels;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
