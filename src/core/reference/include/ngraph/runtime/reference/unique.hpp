// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
/// @brief Represents an element of the input tensor as well as the unique element found by processing the input
template <typename Index_t>
struct Element {
    Element(Index_t idx_) : idx{idx_} {}
    Index_t idx = 0;      // the index of the current element in the original input tensor
    Index_t rev_idx = 0;  // the index of the unique element in the output tensor
    int64_t count = 0;    // the number of occurrences of the current element in the original input tensor
};

template <typename Data_t, typename Index_t>
void unique(Data_t* out_unique_elements,
            Index_t* out_indices,
            Index_t* out_rev_indices,
            int64_t* out_counts,
            const Data_t* data,
            const Shape& data_shape,
            const std::unique_ptr<int64_t> axis,
            const bool sorted) {
    if (data_shape.size() == 0) {
        out_unique_elements[0] = data[0];
        out_indices[0] = 0;
        out_rev_indices[0] = 0;
        out_counts[0] = 1;
    } else if (data_shape.size() == 1) {
        const auto elems_count = shape_size(data_shape);

        std::vector<Element<Index_t>> data_references;
        data_references.reserve(elems_count);

        for (int i = 0; i < elems_count; ++i) {
            data_references.emplace_back(i);
        }

        std::sort(std::begin(data_references),
                  std::end(data_references),
                  [&data](const Element<Index_t>& lhs, const Element<Index_t>& rhs) {
                      return *(data + lhs.idx) < *(data + rhs.idx);
                  });
    } else {
        throw std::runtime_error("Not implemented yet");
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
