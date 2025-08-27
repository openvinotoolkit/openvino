// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.hpp"

/// Read a 4-bit value from packed int4 array
uint8_t details::read_4b(const uint8_t* data, size_t r, size_t c, size_t cols) {
    size_t idx = r * cols + c;
    uint8_t byte = data[idx / 2];
    return (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
}

// Write a 4-bit value to packed int4 array
void details::write_4b(uint8_t* data, uint8_t val, size_t r, size_t c, size_t cols) {
    size_t idx = r * cols + c;
    // uint8_t* byte = data + idx;
    uint8_t& byte = data[idx / 2];
    if (idx % 2 == 0) {
        byte = (byte & 0xF0) | (val & 0x0F);
    } else {
        byte = (byte & 0x0F) | ((val & 0x0F) << 4);
    }
}

::testing::AssertionResult details::ArraysMatch(const std::vector<int8_t>& actual,
                                                const std::vector<int8_t>& expected) {
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << "Size mismatch: actual.size()=" << actual.size() << ", expected.size()=" << expected.size();
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            return ::testing::AssertionFailure()
                   << "Mismatch at index " << i << ": actual=" << static_cast<int>(actual[i])
                   << ", expected=" << static_cast<int>(expected[i]);
        }
    }
    return ::testing::AssertionSuccess();
}

std::vector<size_t> details::compute_strides(const std::vector<size_t>& dims) {
    std::vector<size_t> strides(dims.size());
    size_t stride = 1;
    for (size_t i = dims.size(); i > 0; --i) {
        strides[i - 1] = stride;
        stride *= dims[i - 1];
    }
    return strides;
}

std::vector<size_t> details::reorder(const std::vector<size_t>& dims, const std::vector<size_t>& order) {
    std::vector<size_t> new_dims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
        new_dims[i] = dims[order[i]];

    return new_dims;
}

::testing::internal::ParamGenerator<typename std::vector<ShapesInitializer>::value_type> details::ShapesIn(
    const std::vector<ShapesInitializer>& container) {
    return ::testing::ValuesIn(container.begin(), container.end());
}