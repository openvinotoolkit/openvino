// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/// Unit tests for ov::util::ParallelMemStreamBuf.
///
/// Goals:
///  1. Verify byte-exact correctness for small reads (memcpy path, threshold=SIZE_MAX).
///  2. Verify byte-exact correctness for the parallel memcpy path (threshold=1).
///  3. Verify correct behaviour for non-zero logical start (pointer offset within
///     a larger allocation).
///  4. Verify seekoff / seekpos for all three seek directions (beg, cur, end).
///  5. Verify underflow() / get() path (peeking + char-by-char consumption).
///  6. Verify uflow() advances the internal cursor correctly.
///  7. Verify showmanyc() / in_avail() reports remaining bytes accurately.
///  8. Verify boundary conditions: out-of-range seek, read beyond end.
///  9. Verify mixed underflow + bulk read: drain underflow first, then xsgetn.
/// 10. Verify large parallel memcpy (>= DEFAULT_THRESHOLD on any ≥2-core machine).
///
/// The "parallel dispatch" tests use buffers large enough that
///   size / MIN_CHUNK_SIZE (2 MB) >= 2
/// on any machine, so num_chunks > 1 whenever hardware parallelism exists.
/// With threshold=1 the parallel_copy() branch is taken for every xsgetn call.

#include "openvino/util/parallel_mem_streambuf.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <vector>

namespace ov::test {

// ---------------------------------------------------------------------------
// Helper: fill a vector with a deterministic pattern unique per byte index.
// Using prime modulus 251 so the period is never aligned with any power-of-two
// chunk or page size.
// ---------------------------------------------------------------------------
namespace {
void fill_pattern(std::vector<uint8_t>& buf, size_t start_index = 0) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<uint8_t>((start_index + i) % 251u);
    }
}
}  // namespace

// ---------------------------------------------------------------------------
// 1. Small read – threshold=SIZE_MAX forces the single memcpy path.
//    Verifies the basic xsgetn wire-up and cursor advance.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, FullReadSingleMemcpyPath) {
    constexpr size_t k_size = 4 * 1024;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    std::vector<uint8_t> got(k_size);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, src);
}

// ---------------------------------------------------------------------------
// 2. threshold=1 forces parallel_copy() to be called on every bulk read.
//    Verifies byte-exact output regardless of which code path is taken.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, FullReadParallelMemcpyPath) {
    constexpr size_t k_size = 8 * 1024;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(k_size);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, src);
}

// ---------------------------------------------------------------------------
// 3. Non-zero logical start: construct on a sub-span of a larger allocation.
//    Bytes before the sub-span pointer must never appear in reads.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, NonZeroPointerOffset) {
    constexpr size_t k_prefix_size = 512;
    constexpr size_t k_payload_size = 2 * 1024;
    std::vector<uint8_t> backing(k_prefix_size + k_payload_size);
    // Fill the entire backing buffer; prefix bytes = 0xFF (should never be read)
    std::fill(backing.begin(), backing.begin() + k_prefix_size, 0xFFu);
    std::vector<uint8_t> payload(k_payload_size);
    fill_pattern(payload);
    std::memcpy(backing.data() + k_prefix_size, payload.data(), k_payload_size);

    const char* payload_ptr = reinterpret_cast<const char*>(backing.data() + k_prefix_size);
    util::ParallelMemStreamBuf buf(payload_ptr, k_payload_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(k_payload_size);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(k_payload_size)));
    EXPECT_EQ(got, payload);
}

// ---------------------------------------------------------------------------
// 4. Multiple consecutive partial reads consume bytes in order.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, ChunkedReads) {
    constexpr size_t k_size = 8 * 1024;
    constexpr size_t k_chunk = 1000;  // intentionally not a power-of-2
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(k_size);
    size_t offset = 0;
    while (offset < k_size) {
        const size_t n = std::min(k_chunk, k_size - offset);
        ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data() + offset), static_cast<std::streamsize>(n)));
        offset += n;
    }
    EXPECT_EQ(got, src);
}

// ---------------------------------------------------------------------------
// 5. underflow() + uflow() – char-by-char consumption via stream.get().
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, CharByCharRead) {
    constexpr size_t k_size = 200;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    std::vector<uint8_t> got;
    got.reserve(k_size);
    int ch;
    while ((ch = stream.get()) != std::char_traits<char>::eof()) {
        got.push_back(static_cast<uint8_t>(ch));
    }
    ASSERT_EQ(got.size(), k_size);
    EXPECT_EQ(got, src);
}

// ---------------------------------------------------------------------------
// 6. seekg(pos, beg) then read returns bytes at that logical position.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, SeekFromBeginning) {
    constexpr size_t k_size = 1024;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr std::streamoff k_seek_pos = 300;
    constexpr size_t k_read_len = 20;
    stream.seekg(k_seek_pos, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> got(k_read_len);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(k_read_len)));

    std::vector<uint8_t> expected(src.begin() + k_seek_pos, src.begin() + k_seek_pos + k_read_len);
    EXPECT_EQ(got, expected);
}

// ---------------------------------------------------------------------------
// 7. seekg(off, cur) – relative forward seek after an initial read.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, SeekFromCurrent) {
    constexpr size_t k_size = 1024;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr size_t k_first_read = 100;
    constexpr std::streamoff k_skip = 150;
    constexpr size_t k_second_read = 30;

    std::vector<uint8_t> first(k_first_read);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(first.data()), static_cast<std::streamsize>(k_first_read)));
    EXPECT_EQ(first, std::vector<uint8_t>(src.begin(), src.begin() + k_first_read));

    stream.seekg(k_skip, std::ios::cur);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> second(k_second_read);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(second.data()), static_cast<std::streamsize>(k_second_read)));

    const size_t expected_start = k_first_read + static_cast<size_t>(k_skip);
    std::vector<uint8_t> expected_slice(src.begin() + expected_start, src.begin() + expected_start + k_second_read);
    EXPECT_EQ(second, expected_slice);
}

// ---------------------------------------------------------------------------
// 8. seekg(off, end) – backward seek from the end.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, SeekFromEnd) {
    constexpr size_t k_size = 1024;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr std::streamoff k_from_end = 48;
    stream.seekg(-k_from_end, std::ios::end);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> got(static_cast<size_t>(k_from_end));
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), k_from_end));

    std::vector<uint8_t> expected(src.end() - k_from_end, src.end());
    EXPECT_EQ(got, expected);
}

// ---------------------------------------------------------------------------
// 9. seekg(0, end) then tellg() must equal the buffer size.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, TellgAtEnd) {
    constexpr size_t k_size = 512;
    std::vector<uint8_t> src(k_size, 0xAAu);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    stream.seekg(0, std::ios::end);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), k_size);
}

// ---------------------------------------------------------------------------
// 10. tellg() reflects the current position accurately after mixed reads and
//     seeks.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, TellgIsConsistent) {
    constexpr size_t k_size = 512;
    std::vector<uint8_t> src(k_size, 0xBBu);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    EXPECT_EQ(stream.tellg(), std::streampos(0));

    std::vector<char> tmp(100);
    stream.read(tmp.data(), 100);
    EXPECT_EQ(stream.tellg(), std::streampos(100));

    for (int i = 0; i < 10; ++i) {
        stream.get();
    }
    EXPECT_EQ(stream.tellg(), std::streampos(110));

    stream.seekg(200, std::ios::beg);
    EXPECT_EQ(stream.tellg(), std::streampos(200));
}

// ---------------------------------------------------------------------------
// 11. Out-of-range seek returns pos_type(-1).
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, OutOfRangeSeekFails) {
    constexpr size_t k_size = 64;
    std::vector<uint8_t> src(k_size, 0x55u);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    // Seek before the beginning
    const auto pos = stream.seekg(-1, std::ios::beg).tellg();
    EXPECT_EQ(pos, std::streampos(-1));
}

// ---------------------------------------------------------------------------
// 12. Partial read at EOF: requesting more bytes than remain must return false,
//     set stream.eof(), and deliver only the bytes that were available via
//     gcount().
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, ReadAtEof) {
    constexpr size_t k_size = 80;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Consume all but the last 10 bytes
    std::vector<uint8_t> discard(k_size - 10);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(discard.data()), static_cast<std::streamsize>(k_size - 10)));

    // Request 20 bytes when only 10 remain
    std::vector<uint8_t> tail(20, 0xFFu);
    const bool ok = static_cast<bool>(stream.read(reinterpret_cast<char*>(tail.data()), 20));
    EXPECT_FALSE(ok);
    EXPECT_TRUE(stream.eof());
    ASSERT_EQ(stream.gcount(), 10);
    EXPECT_TRUE(std::equal(tail.begin(), tail.begin() + 10, src.end() - 10));
}

// ---------------------------------------------------------------------------
// 13. showmanyc() / in_avail() reports the correct number of remaining bytes
//     and -1 when the buffer is exhausted.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, ShowmanycReflectsRemainingBytes) {
    constexpr size_t k_size = 256;
    std::vector<uint8_t> src(k_size, 0x77u);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size));

    std::vector<char> tmp(100);
    stream.read(tmp.data(), 100);
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size - 100));

    // Consume remaining bytes
    std::vector<char> rest(k_size - 100);
    stream.read(rest.data(), static_cast<std::streamsize>(k_size - 100));
    EXPECT_EQ(stream.rdbuf()->in_avail(), -1);
}

// ---------------------------------------------------------------------------
// 14. Mixed underflow + bulk read: first consume bytes char-by-char, then
//     switch to stream.read() for the tail.
//     (ParallelMemStreamBuf has no internal buffer to drain; the transition
//      tests that m_current advances cleanly across both call paths.)
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, MixedCharAndBulkRead) {
    constexpr size_t k_size = 1024;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Read 8 bytes individually
    for (int i = 0; i < 8; ++i) {
        const int ch = stream.get();
        ASSERT_NE(ch, std::char_traits<char>::eof());
        EXPECT_EQ(static_cast<uint8_t>(ch), src[static_cast<size_t>(i)]);
    }

    // Read the rest via bulk read
    std::vector<uint8_t> rest(k_size - 8);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(rest.data()), static_cast<std::streamsize>(k_size - 8)));
    EXPECT_EQ(rest, std::vector<uint8_t>(src.begin() + 8, src.end()));
}

// ---------------------------------------------------------------------------
// 15. Seek back to position 0 and re-read the full buffer; verifies that the
//     internal cursor is properly reset.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, SeekToZeroAndReread) {
    constexpr size_t k_size = 1024;
    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    // First full read
    std::vector<uint8_t> first(k_size);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(first.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(first, src);

    // Seek back and re-read
    stream.clear();
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> second(k_size);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(second.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(second, src);
}

// ---------------------------------------------------------------------------
// 16. PARALLEL PATH CORRECTNESS – large buffer that exceeds the 2 MB minimum
//     chunk size on any ≥2-core machine so that num_chunks > 1 and the
//     ov::parallel_for() dispatch in parallel_copy() actually fires.
//     Uses threshold=1 to ensure parallel_copy() is invoked.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, ParallelDispatchFullReadCorrectness) {
    // 2 * hw_threads * 2 MB + 1: guarantees num_chunks >= 2 on hardware that has
    // at least 2 threads, since MIN_CHUNK_SIZE inside parallel_copy is 2 MB.
    const size_t k_max_hw = 16;
    const size_t hw_raw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t hw = hw_raw > k_max_hw ? k_max_hw : hw_raw;
    const size_t k_size = 2u * hw * 2u * 1024u * 1024u + 1u;

    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(k_size);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, src) << "Parallel memcpy produced incorrect data";
}

// ---------------------------------------------------------------------------
// 17. PARALLEL PATH with mid-stream seek: read the first half in parallel,
//     seek back to 0, read everything again.  Verifies that the cursor reset
//     is correct after a parallel bulk read.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, ParallelDispatchSeekAndReread) {
    const size_t k_max_hw = 16;
    const size_t hw_raw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t hw = hw_raw > k_max_hw ? k_max_hw : hw_raw;
    const size_t k_size = 2u * hw * 2u * 1024u * 1024u;

    std::vector<uint8_t> src(k_size);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    // Read first half
    const size_t k_half = k_size / 2;
    std::vector<uint8_t> first_half(k_half);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(first_half.data()), static_cast<std::streamsize>(k_half)));
    EXPECT_TRUE(std::equal(first_half.begin(), first_half.end(), src.begin()));

    // Seek back and read the full buffer
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> full(k_size);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(full.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(full, src) << "Full read after seek produced incorrect data";
}

}  // namespace ov::test
