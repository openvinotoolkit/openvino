// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/mmap_test_utils.hpp"

namespace ov::test::utils {

namespace {
constexpr uint8_t k_prime_modulus = 251;
}  // namespace

std::vector<uint8_t> make_prime_pattern(size_t size) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(i % k_prime_modulus);
    }
    return data;
}

}  // namespace ov::test::utils
