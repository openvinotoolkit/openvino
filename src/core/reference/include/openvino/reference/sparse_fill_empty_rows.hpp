// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>
#include <iostream>

#include "openvino/core/shape.hpp"

namespace ov::reference {

template <typename T, typename T_IDX>
void sparse_fill_empty_rows(const T* values,
                            const size_t values_size,
                            const T_IDX* dense_shape,
                            const T_IDX* indices,
                            const T default_value,
                            T_IDX* output_indices,
                            T* output_values,
                            bool* empty_row_indicator) {
    // Debug print input parameters
    std::cout << "DEBUG Reference sparse_fill_empty_rows called with:" << std::endl;
    std::cout << "  values_size: " << values_size << std::endl;
    std::cout << "  dense_shape: [" << dense_shape[0] << ", " << dense_shape[1] << "]" << std::endl;
    std::cout << "  default_value: " << default_value << std::endl;
    
    // Print first few indices if available
    if (values_size > 0 && indices != nullptr) {
        std::cout << "  indices sample: [" << indices[0] << ", " << indices[1] << "]";
        if (values_size > 1) {
            std::cout << ", [" << indices[2] << ", [" << indices[3] << "]";
        }
        std::cout << std::endl;
    } else {
        std::cout << "  indices: empty" << std::endl;
    }
    
    const auto num_rows = dense_shape[0];

    std::unordered_set<T_IDX> existing_rows;
    for (size_t i = 0, idx = 0; i < values_size; i++, idx += 2) {
        existing_rows.insert(indices[idx]);
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

    // Vector of pairs containing ((row, column), source_index) for
    // both existing values and new empty rows to be added
    const size_t total_rows = values_size + empty_rows.size();
    std::vector<std::pair<std::pair<T_IDX, T_IDX>, size_t>> row_col_pairs(total_rows);

    // Add existing values and then empty rows
    for (size_t i = 0, idx = 0; i < values_size; i++, idx += 2) {
        row_col_pairs[i] = {{indices[idx], indices[idx + 1]}, i};
    }
    for (size_t i = 0; i < empty_rows.size(); i++) {
        row_col_pairs[values_size + i] = {{empty_rows[i], 0}, values_size + i};
    }

    std::sort(row_col_pairs.begin(), row_col_pairs.end(), [](const auto& a, const auto& b) {
        if (a.first.first != b.first.first) {
            return a.first.first < b.first.first;
        }
        return a.first.second < b.first.second;
    });

    for (size_t i = 0, out_idx = 0; i < total_rows; i++, out_idx += 2) {
        const auto& [row_col, src_idx] = row_col_pairs[i];
        const auto& [row, col] = row_col;

        output_indices[out_idx] = row;
        output_indices[out_idx + 1] = col;

        if (src_idx < values_size) {
            output_values[i] = values[src_idx];
        } else {
            output_values[i] = default_value;
        }
    }

    // Add debug prints for all output tensors
    std::cout << "DEBUG Reference output tensors:" << std::endl;
    
    // Print output_indices
    std::cout << "  output_indices: [";
    for (size_t i = 0; i < total_rows * 2; i += 2) {
        if (i > 0) std::cout << ", ";
        std::cout << "[" << output_indices[i] << ", " << output_indices[i+1] << "]";
    }
    std::cout << "]" << std::endl;
    
    // Print output_values
    std::cout << "  output_values: [";
    for (size_t i = 0; i < total_rows; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << output_values[i];
    }
    std::cout << "]" << std::endl;
    
    // Print empty_row_indicator
    std::cout << "  empty_row_indicator: [";
    for (T_IDX i = 0; i < num_rows; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << (empty_row_indicator[i] ? "true" : "false");
    }
    std::cout << "]" << std::endl;
}

}  // namespace ov::reference
