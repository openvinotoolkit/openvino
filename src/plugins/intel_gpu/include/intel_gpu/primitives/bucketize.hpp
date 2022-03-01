// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>

#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Bucketize operation bucketizes the input based on boundaries.
/// @details Bucketize operation computes a bucket index for each element from the first input and outputs a tensor of
/// the first input shape. Buckets are defined with boundaries from the second input. For example, if the first input
/// tensor is [[3, 50], [10, -1]] and the second input is [0, 5, 10] with included right bound, the output will be
///  [[1, 3], [2, 0]]. Reference: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/bucketize

struct bucketize : public primitive_base<bucketize> {
    CLDNN_DECLARE_PRIMITIVE(bucketize)

    /// @brief Constructs bucketize primitive
    /// @param id This primitive id
    /// @param inputs Inputs for primitive id
    /// @param output_type The output tensor type
    /// @param with_right_bound Indicates whether bucket includes the right or the left edge of interval
    bucketize(const primitive_id& id,
              const primitive_id& input,
              const primitive_id& buckets,
              data_types output_type = data_types::i64,
              const bool with_right_bound = true,
              const primitive_id& ext_prim_id = "",
              const padding& output_padding = padding())
        : primitive_base(id, {input, buckets}, ext_prim_id, output_padding),
          output_type(output_type),
          with_right_bound(with_right_bound) {}

    data_types output_type;
    bool with_right_bound;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
