/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include <algorithm>
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Finds the index of the k max values of input.
/// @details Returns indices in f32, because we currently does not support int32 data type.
/// We use f32, as bigger indices could not fit in smaller data types.
/// If you want to use output as indices outside of network (inside just use lookup table primitive),
/// you will need to firstly cast it to int (look into tests for example).
struct arg_max_min : public primitive_base<arg_max_min> {
    CLDNN_DECLARE_PRIMITIVE(arg_max_min)

    /// @brief Enum type to specify axis to return values from.
    enum out_type {
        max,
        min,
    };

    /// @brief Enum type to specify axis to maximize/minimize along.
    enum axis_name { batch, feature, x, y, z, xyf };

    /// @brief Enum type to specify sort by values or indices.
    enum sort_type { sort_by_values, sort_by_indices };

    /// @brief Constructs arg_max_min primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param out_type Type of output - max or mix.
    /// @param top_k Number of indices to output.
    /// @param axis Axis to maximize/minimize along.
    arg_max_min(const primitive_id& id,
                const std::vector<primitive_id>& input,
                out_type output_type,
                uint32_t top_k = 1,
                axis_name axis = axis_name::xyf,
                sort_type sort = sort_type::sort_by_values,
                bool values_first = false,
                const padding& output_padding = padding(),
                data_types output_data_type = data_types::f32)
        : primitive_base(id, {input}, output_padding, optional_data_type {output_data_type}),
          top_k(top_k),
          output_type(output_type),
          axis(axis),
          sort(sort),
          with_axis(axis == axis_name::xyf ? false : true),
          values_first(values_first) {}

    /// @brief Number of indices to output.
    uint32_t top_k;
    /// @brief Type of output - max or mix.
    out_type output_type;
    /// @brief Axis to maximize/minimize along. If not set, maximize the flattened trailing dimensions for each index of the batch dimension.
    axis_name axis;
    /// @brief Type of sorting - by values or indices.
    sort_type sort;
    /// @brief Indicates that the primitive has user defined axis to maximize/minimize along;
    bool with_axis;
    /// @brief Sets output order: if True than first output contains values and second (optional) - indices.
    bool values_first;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
