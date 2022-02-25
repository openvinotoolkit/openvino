// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/utils/fft_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>

#include "ngraph/check.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace fft_common {
std::vector<int64_t> reverse_shape(const ngraph::Shape& shape) {
    size_t complex_data_rank = shape.size() - 1;

    std::vector<int64_t> reversed_shape(complex_data_rank);
    for (size_t i = 0; i < complex_data_rank; ++i) {
        reversed_shape[i] = static_cast<int64_t>(shape[complex_data_rank - i - 1]);
    }
    return reversed_shape;
}

std::vector<int64_t> compute_strides(const std::vector<int64_t>& v) {
    std::vector<int64_t> strides(v.size() + 1);
    int64_t stride = 1;
    for (size_t i = 0; i < v.size(); ++i) {
        strides[i] = stride;
        stride *= v[i];
    }
    strides.back() = stride;
    return strides;
}

std::vector<int64_t> coords_from_index(int64_t index, const std::vector<int64_t>& strides) {
    int64_t num_of_axes = static_cast<int64_t>(strides.size()) - 1;
    if (num_of_axes == 0) {
        return std::vector<int64_t>{};
    }
    std::vector<int64_t> coords(num_of_axes);
    int64_t curr = index;
    for (int64_t j = num_of_axes - 1; j >= 1; --j) {
        coords[j] = curr / strides[j];
        curr %= strides[j];
    }
    coords[0] = curr;
    return coords;
}

int64_t offset_from_coords_and_strides(const std::vector<int64_t>& coords, const std::vector<int64_t>& strides) {
    int64_t offset = 0;
    int64_t num_of_axes = coords.size();
    for (int64_t i = 0; i < num_of_axes; ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}
}  // namespace fft_common
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
