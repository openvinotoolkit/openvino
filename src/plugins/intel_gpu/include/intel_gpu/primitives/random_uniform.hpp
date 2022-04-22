// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief RandomUniform-8 primitive
/// @details
struct random_uniform : public primitive_base<random_uniform> {
    CLDNN_DECLARE_PRIMITIVE(random_uniform)

    /**
     * Construct Random Uniform privitive.
     * @param id primitive id
     * @param inputs inputs parameters ids
     * @param data_type output values data type
     * @param global_seed, op_seed random uniform seed attributes
     * @param output_shape output data shape
     * @param output_format output data shape format
     */
    random_uniform(const primitive_id &id, const std::vector<input_info> &inputs,
                   const data_types &data_type, const uint64_t global_seed,
                   const uint64_t op_seed, const tensor output_shape,
                   const format output_format,
                   const padding &output_padding = padding())
            : primitive_base(id, inputs, "", {output_padding},
                             {optional_data_type{data_type}}),
              global_seed(global_seed),
              op_seed(op_seed),
              output_shape(output_shape),
              output_format(output_format) {}

    const uint64_t global_seed;
    const uint64_t op_seed;
    const tensor output_shape;
    const format output_format;
};

}  // namespace cldnn
