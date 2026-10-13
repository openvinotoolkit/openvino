//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "math.hpp"

#include <cstdint>
#include <limits>

bool intel_npu::vm::checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool intel_npu::vm::checkedAddProduct(uint64_t lhs, uint64_t rhs, uint64_t& accumulator) {
    uint64_t product = 0;
    if (!checkedMultiply(lhs, rhs, product)) {
        return false;
    }
    if (accumulator > std::numeric_limits<uint64_t>::max() - product) {
        return false;
    }
    accumulator += product;
    return true;
}

bool intel_npu::vm::checkedMultiplyNonNegative(int64_t lhs, int64_t rhs, int64_t& result) {
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    uint64_t product = 0;
    if (!checkedMultiply(static_cast<uint64_t>(lhs), static_cast<uint64_t>(rhs), product) ||
        product > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    result = static_cast<int64_t>(product);
    return true;
}
