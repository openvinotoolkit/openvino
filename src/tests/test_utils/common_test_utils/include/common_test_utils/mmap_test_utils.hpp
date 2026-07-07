// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/util/mmap_object.hpp"

namespace ov::test::utils {

// Build a vector<uint8_t> holding the prime-modulo pattern: byte at index i is (i % 251).
// 251 is prime, so the byte period never aligns with any power-of-two page / granularity
// boundary, which makes off-by-page corruption easy to spot.
std::vector<uint8_t> make_prime_pattern(size_t size);

// Snapshot a mapping's bytes into a vector<uint8_t>.
std::vector<uint8_t> read_mapped(ov::MappedMemory& mm);

}  // namespace ov::test::utils
