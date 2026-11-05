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

// Alignment padding inserted before each write() matches the element type's own size.
TEST(ConstantWriterTest, alignment_padding_matches_element_size) {
    std::stringstream bin;
    ov::util::ConstantWriter writer(bin, /*enable_compression=*/false);
    size_t new_size = 0;

    // u8 never needs padding: b starts right after a, at offset 3.
    constexpr std::array<uint8_t, 3> u8_a = {1, 2, 3};
    constexpr std::array<uint8_t, 5> u8_b = {4, 5, 6, 7, 8};
    writer.write(reinterpret_cast<const char*>(u8_a.data()), u8_a.size(), new_size, false, element::u8);
    const auto off_u8_b =
        writer.write(reinterpret_cast<const char*>(u8_b.data()), u8_b.size(), new_size, false, element::u8);
    EXPECT_EQ(off_u8_b, static_cast<int64_t>(u8_a.size()));

    // Stream is at offset 8 now, already 4-aligned: i32 needs no padding either.
    constexpr std::array<uint8_t, 4> i32_data = {};
    const auto off_i32 =
        writer.write(reinterpret_cast<const char*>(i32_data.data()), i32_data.size(), new_size, false, element::i32);
    EXPECT_EQ(off_i32, 8);

    // A 1 B spacer leaves the stream at offset 13: the i64 write must pad 3 B to reach 16.
    constexpr std::array<uint8_t, 1> spacer = {0xAB};
    constexpr std::array<uint8_t, 8> i64_data = {};
    writer.write(reinterpret_cast<const char*>(spacer.data()), spacer.size(), new_size, false, element::u8);
    const auto off_i64 =
        writer.write(reinterpret_cast<const char*>(i64_data.data()), i64_data.size(), new_size, false, element::i64);
    EXPECT_EQ(off_i64, 16);
}

// StreamSerialize embeds the constants blob after an unpadded, arbitrary-length custom-data
// section, so the blob can start at a non-aligned absolute position. Padding must target the
// absolute stream position, not the offset relative to that (possibly unaligned) blob start.
TEST(ConstantWriterTest, padding_targets_absolute_stream_position) {
    std::stringstream bin;
    bin.write("XXXXX", 5);
    const auto blob_offset = static_cast<size_t>(bin.tellp());

    ov::util::ConstantWriter writer(bin, /*enable_compression=*/false);
    size_t new_size = 0;

    constexpr std::array<uint8_t, 1> u8_data = {0xAB};
    constexpr std::array<uint8_t, 4> i32_data = {1, 2, 3, 4};
    writer.write(reinterpret_cast<const char*>(u8_data.data()), u8_data.size(), new_size, false, element::u8);
    const auto off_i32 =
        writer.write(reinterpret_cast<const char*>(i32_data.data()), i32_data.size(), new_size, false, element::i32);

    EXPECT_EQ((blob_offset + static_cast<size_t>(off_i32)) % 4, 0u);
}

}  // namespace ov::test
