// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "openvino/op/util/attr_types.hpp"

#include <algorithm>
#include <vector>

namespace cldnn {

/// @brief Finds the index of the k max values of input.
/// @details Returns indices in f32, because we currently does not support int32 data type.
/// We use f32, as bigger indices could not fit in smaller data types.
/// If you want to use output as indices outside of network (inside just use lookup table primitive),
/// you will need to firstly cast it to int (look into tests for example).
struct arg_max_min : public primitive_base<arg_max_min> {
    CLDNN_DECLARE_PRIMITIVE(arg_max_min)

    arg_max_min() : primitive_base("", {}),
                    mode(ov::op::TopKMode::MAX),
                    top_k(0),
                    axis(0),
                    sort(ov::op::TopKSortType::NONE),
                    values_first(false),
                    stable(false) {}

    /// @brief Constructs arg_max_min primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mode Type of output - max or min.
    /// @param top_k Number of indices to output.
    /// @param axis Axis to maximize/minimize along.
    /// @param sort Type of sorting - by values or indices.
    /// @param stable Controls whether sorting is stable.
    arg_max_min(const primitive_id& id,
                const std::vector<input_info>& inputs,
                ov::op::TopKMode mode,
                uint32_t top_k,
                int64_t axis,
                ov::op::TopKSortType sort = ov::op::TopKSortType::SORT_VALUES,
                bool values_first = false,
                bool stable = false,
                data_types output_data_type = data_types::f32,
                const size_t num_outputs = 1)
        : primitive_base(id, inputs, num_outputs, {optional_data_type{output_data_type}}),
          mode(mode),
          top_k(top_k),
          axis(axis),
          sort(sort),
          values_first(values_first),
          stable(stable) {}

    /// @brief Constructs arg_max_min for top_k parameter
    arg_max_min(const primitive_id& id,
                const input_info& input,
                const input_info& topk_id,
                ov::op::TopKMode mode,
                uint32_t top_k,
                int64_t axis,
                ov::op::TopKSortType sort = ov::op::TopKSortType::SORT_VALUES,
                bool values_first = false,
                bool stable = false,
                data_types output_data_type = data_types::f32,
                const size_t num_outputs = 1)
        : primitive_base(id, {input, topk_id}, num_outputs, {optional_data_type{output_data_type}}),
          mode(mode),
          top_k(top_k),
          axis(axis),
          sort(sort),
          values_first(values_first),
          stable(stable) {}

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
    /// @brief Specifies whether the equivalent elements should maintain their relative order from the input tensor during sorting.
    bool stable;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, mode);
        seed = hash_combine(seed, top_k);
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, sort);
        seed = hash_combine(seed, values_first);
        seed = hash_combine(seed, stable);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const arg_max_min>(rhs);

        return mode == rhs_casted.mode &&
               top_k == rhs_casted.top_k &&
               axis == rhs_casted.axis &&
               sort == rhs_casted.sort &&
               values_first == rhs_casted.values_first &&
               stable == rhs_casted.stable;
    }

    size_t get_output_nums() const {
        return (input_size() == 3 ? 2 : output_size());
    }
    bool has_second_output() const { return get_output_nums() == 2; }
    bool use_multiple_outputs() const { return input_size() != 3; }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<arg_max_min>::save(ob);
        ob << make_data(&mode, sizeof(ov::op::TopKMode));
        ob << top_k;
        ob << axis;
        ob << make_data(&sort, sizeof(ov::op::TopKSortType));
        ob << values_first;
        ob << stable;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<arg_max_min>::load(ib);
        ib >> make_data(&mode, sizeof(ov::op::TopKMode));
        ib >> top_k;
        ib >> axis;
        ib >> make_data(&sort, sizeof(ov::op::TopKSortType));
        ib >> values_first;
        ib >> stable;
    }
};
}  // namespace cldnn
