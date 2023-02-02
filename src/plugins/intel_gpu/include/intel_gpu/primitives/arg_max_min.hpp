// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include "openvino/op/util/attr_types.hpp"

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

    /// @brief Constructs arg_max_min primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mode Type of output - max or min.
    /// @param top_k Number of indices to output.
    /// @param axis Axis to maximize/minimize along.
    /// @param sort Type of sorting - by values or indices.
    arg_max_min(const primitive_id& id,
                const std::vector<input_info>& inputs,
                ov::op::TopKMode mode,
                uint32_t top_k,
                int64_t axis,
                ov::op::TopKSortType sort = ov::op::TopKSortType::SORT_VALUES,
                bool values_first = false,
                const padding& output_padding = padding(),
                data_types output_data_type = data_types::f32,
                const size_t num_outputs = 1)
        : primitive_base(id, inputs, {output_padding}, {optional_data_type{output_data_type}}, num_outputs),
          mode(mode),
          top_k(top_k),
          axis(axis),
          sort(sort),
          values_first(values_first) {}

    /// @brief Type of output - max or min.
    ov::op::TopKMode mode;
    /// @brief Number of indices to output.
    uint32_t top_k;
    /// @brief Axis to maximize/minimize along. If not set, maximize the flattened trailing dimensions for each index of the batch dimension.
    int64_t axis;
    /// @brief Type of sorting - by values or indices.
    ov::op::TopKSortType sort;
    /// @brief Sets output order: if True than first output contains values and second (optional) - indices.
    bool values_first;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
