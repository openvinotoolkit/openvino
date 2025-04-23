// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov::reference {

template <typename T, typename T_IDX>
void sparse_fill_empty_rows(const T* values,
                           const size_t values_size,
                           const T_IDX* dense_shape,
                           const T_IDX* indices,
                           const size_t indices_size,
                           const T default_value,
                           T_IDX* output_indices,
                           T* output_values,
                           bool* empty_row_indicator) {
    const T_IDX num_rows = dense_shape[0];
    const size_t num_entries = values_size;

    std::unordered_set<T_IDX> existing_rows;
    for (size_t i = 0; i < num_entries; i++) {
        // Only 2D inputs are supported, so every second idx is a row idx
        existing_rows.insert(indices[i * 2]);
    }

    std::vector<T_IDX> empty_rows;
    empty_rows.reserve(num_rows - existing_rows.size());
    for (T_IDX i = 0; i < num_rows; i++) {
        const bool is_empty = (existing_rows.find(i) == existing_rows.end());
        empty_row_indicator[i] = is_empty;
        if (is_empty) {
            empty_rows.push_back(i);
        }
    }

    // Copy all existing entries
    size_t output_idx = 0;
    for (size_t i = 0; i < num_entries; i++) {
        output_indices[output_idx * 2] = indices[i * 2];
        output_indices[output_idx * 2 + 1] = indices[i * 2 + 1];
        output_values[output_idx] = values[i];
        output_idx++;
    }

    // Add entries for empty rows, first column
    for (const auto& row : empty_rows) {
        output_indices[output_idx * 2] = row;
        output_indices[output_idx * 2 + 1] = 0;
        output_values[output_idx] = default_value;
        output_idx++;
    }

    // Sort the output in row-major order (first by row, then by column)
    std::vector<size_t> permutation(output_idx);
    for (size_t i = 0; i < output_idx; i++) {
        permutation[i] = i;
    }
    std::stable_sort(permutation.begin(), permutation.end(), [&](size_t a, size_t b) {
        T_IDX row_a = output_indices[a * 2];
        T_IDX col_a = output_indices[a * 2 + 1];
        T_IDX row_b = output_indices[b * 2];
        T_IDX col_b = output_indices[b * 2 + 1];

        if (row_a != row_b) {
            return row_a < row_b;
        }
        return col_a < col_b;
    });
    std::vector<T_IDX> sorted_indices(output_idx * 2);
    std::vector<T> sorted_values(output_idx);

    for (size_t i = 0; i < output_idx; i++) {
        size_t src_idx = permutation[i];
        sorted_indices[i * 2] = output_indices[src_idx * 2];
        sorted_indices[i * 2 + 1] = output_indices[src_idx * 2 + 1];
        sorted_values[i] = output_values[src_idx];
    }

    std::copy(sorted_indices.begin(), sorted_indices.end(), output_indices);
    std::copy(sorted_values.begin(), sorted_values.end(), output_values);
}

}  // namespace ov::reference
