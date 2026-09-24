// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/xml_util/constant_writer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <sstream>
#include <vector>

#include "openvino/core/visibility.hpp"

namespace ov::test {

// Tests for ConstantWriter's deduplication size guard against hash collisions between different-sized buffers.

// Precomputed collision: k_large starts with k_small and both hash to the same value. Valid only for
// the x86-64 CRC-64 hash, so the tests are x86-64 only and assert the collision to catch hash changes.
#if defined(OPENVINO_ARCH_X86_64)
namespace {
constexpr std::array<uint8_t, 16> k_small =
    {0x07, 0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f, 0xa0, 0xb1, 0xc2, 0xd3, 0xe4, 0xf5, 0x06};
constexpr std::array<uint8_t, 24> k_large = {0x07, 0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f, 0xa0, 0xb1, 0xc2,
                                             0xd3, 0xe4, 0xf5, 0x06, 0x2b, 0xa4, 0x34, 0x82, 0x0b, 0x04, 0xb1, 0x2a};

// Writes compile-time byte data through ConstantWriter, reinterpreting it as the char buffer the API expects.
template <size_t N>
ov::util::ConstantWriter::FilePosition write_bytes(ov::util::ConstantWriter& writer,
                                                   const std::array<uint8_t, N>& data,
                                                   size_t& new_size) {
    return writer.write(reinterpret_cast<const char*>(data.data()), data.size(), new_size);
}

// Surfaces the internal per-buffer hash through the public get_data_hash(): a fresh writer combines it
// as u64_hash_combine(0, hash), which is one-to-one, so equal results imply equal hashes.
template <size_t N>
uint64_t const_write_hash(const std::array<uint8_t, N>& data) {
    std::stringstream bin;
    ov::util::ConstantWriter writer(bin, /*enable_compression=*/true);
    size_t new_size = 0;
    write_bytes(writer, data, new_size);
    return writer.get_data_hash();
}
}  // namespace

TEST(ConstantWriterTest, size_mismatch_is_not_deduplicated) {
    ASSERT_EQ(const_write_hash(k_small), const_write_hash(k_large))
        << "hardcoded buffers no longer collide; regenerate them for the current hash";

    std::stringstream bin;
    ov::util::ConstantWriter writer(bin, /*enable_compression=*/true);
    size_t new_size = 0;
    const auto off_large = write_bytes(writer, k_large, new_size);
    const auto off_small = write_bytes(writer, k_small, new_size);

    EXPECT_NE(off_small, off_large) << "a shorter constant must not be deduplicated onto a longer colliding one";
    EXPECT_EQ(static_cast<size_t>(bin.tellp()), k_large.size() + k_small.size()) << "both constants must be written";
}

TEST(ConstantWriterTest, hash_collision_with_larger_current_buffer_no_oob) {
    ASSERT_EQ(const_write_hash(k_small), const_write_hash(k_large))
        << "hardcoded buffers no longer collide; regenerate them for the current hash";

    std::stringstream bin;
    ov::util::ConstantWriter writer(bin, /*enable_compression=*/true);
    size_t new_size = 0;
    const auto off_small = write_bytes(writer, k_small, new_size);
    const auto off_large = write_bytes(writer, k_large, new_size);

    EXPECT_NE(off_large, off_small) << "a longer constant must not be deduplicated onto a shorter colliding one";
    EXPECT_EQ(static_cast<size_t>(bin.tellp()), k_small.size() + k_large.size()) << "both constants must be written";
}
#endif  // OPENVINO_ARCH_X86_64

TEST(ConstantWriterTest, identical_constants_are_deduplicated) {
    const std::vector<char> a(128, char{0x3C});
    const std::vector<char> b(128, char{0x3C});

    std::stringstream bin;
    ov::util::ConstantWriter writer(bin, /*enable_compression=*/true);
    size_t new_size = 0;
    const auto off_a = writer.write(a.data(), a.size(), new_size);
    const auto off_b = writer.write(b.data(), b.size(), new_size);

    EXPECT_EQ(off_a, off_b) << "identical constants must still be deduplicated";
    EXPECT_EQ(static_cast<size_t>(bin.tellp()), a.size()) << "duplicate constant must not be re-written";
}

}  // namespace ov::test
