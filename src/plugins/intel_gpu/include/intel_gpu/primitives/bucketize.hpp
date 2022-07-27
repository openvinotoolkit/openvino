// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Bucketize operation bucketizes the input based on boundaries.
struct bucketize : primitive_base<bucketize> {
    CLDNN_DECLARE_PRIMITIVE(bucketize)

    /// @brief Constructs bucketize primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param output_type Output tensor type.
    /// @param with_right_bound Indicates whether bucket includes the right or the left edge of interval.
    bucketize(const primitive_id& id,
              const std::vector<primitive_id>& inputs,
              data_types output_type = data_types::i64,
              bool with_right_bound = true,
              const primitive_id& ext_prim_id = {},
              const padding& output_padding = {})
        : primitive_base(id, inputs, ext_prim_id, output_padding, optional_data_type(output_type)),
          with_right_bound(with_right_bound) {}

    bool with_right_bound;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
