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
TEST(ParallelMemStreamBufTest, FullRead_SingleMemcpyPath) {
    constexpr size_t kSize = 4 * 1024;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kSize)));
    EXPECT_EQ(got, src);
}

// ---------------------------------------------------------------------------
// 2. threshold=1 forces parallel_copy() to be called on every bulk read.
//    Verifies byte-exact output regardless of which code path is taken.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, FullRead_ParallelMemcpyPath) {
    constexpr size_t kSize = 8 * 1024;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kSize)));
    EXPECT_EQ(got, src);
}

// ---------------------------------------------------------------------------
// 3. Non-zero logical start: construct on a sub-span of a larger allocation.
//    Bytes before the sub-span pointer must never appear in reads.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, NonZeroPointerOffset) {
    constexpr size_t kPrefixSize = 512;
    constexpr size_t kPayloadSize = 2 * 1024;
    std::vector<uint8_t> backing(kPrefixSize + kPayloadSize);
    // Fill the entire backing buffer; prefix bytes = 0xFF (should never be read)
    std::fill(backing.begin(), backing.begin() + kPrefixSize, 0xFFu);
    std::vector<uint8_t> payload(kPayloadSize);
    fill_pattern(payload);
    std::memcpy(backing.data() + kPrefixSize, payload.data(), kPayloadSize);

    const char* payload_ptr = reinterpret_cast<const char*>(backing.data() + kPrefixSize);
    util::ParallelMemStreamBuf buf(payload_ptr, kPayloadSize, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kPayloadSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kPayloadSize)));
    EXPECT_EQ(got, payload);
}

// ---------------------------------------------------------------------------
// 4. Multiple consecutive partial reads consume bytes in order.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, ChunkedReads) {
    constexpr size_t kSize = 8 * 1024;
    constexpr size_t kChunk = 1000;  // intentionally not a power-of-2
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kSize);
    size_t offset = 0;
    while (offset < kSize) {
        const size_t n = std::min(kChunk, kSize - offset);
        ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data() + offset), static_cast<std::streamsize>(n)));
        offset += n;
    }
    EXPECT_EQ(got, src);
}

// ---------------------------------------------------------------------------
// 5. underflow() + uflow() – char-by-char consumption via stream.get().
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, CharByCharRead) {
    constexpr size_t kSize = 200;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    std::vector<uint8_t> got;
    got.reserve(kSize);
    int ch;
    while ((ch = stream.get()) != std::char_traits<char>::eof()) {
        got.push_back(static_cast<uint8_t>(ch));
    }
    ASSERT_EQ(got.size(), kSize);
    EXPECT_EQ(got, src);
}

// ---------------------------------------------------------------------------
// 6. seekg(pos, beg) then read returns bytes at that logical position.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, SeekFromBeginning) {
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr std::streamoff kSeekPos = 300;
    constexpr size_t kReadLen = 20;
    stream.seekg(kSeekPos, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> got(kReadLen);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kReadLen)));

    std::vector<uint8_t> expected(src.begin() + kSeekPos, src.begin() + kSeekPos + kReadLen);
    EXPECT_EQ(got, expected);
}

// ---------------------------------------------------------------------------
// 7. seekg(off, cur) – relative forward seek after an initial read.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, SeekFromCurrent) {
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr size_t kFirstRead = 100;
    constexpr std::streamoff kSkip = 150;
    constexpr size_t kSecondRead = 30;

    std::vector<uint8_t> first(kFirstRead);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(first.data()), static_cast<std::streamsize>(kFirstRead)));
    EXPECT_EQ(first, std::vector<uint8_t>(src.begin(), src.begin() + kFirstRead));

    stream.seekg(kSkip, std::ios::cur);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> second(kSecondRead);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(second.data()), static_cast<std::streamsize>(kSecondRead)));

    const size_t expected_start = kFirstRead + static_cast<size_t>(kSkip);
    std::vector<uint8_t> expected_slice(src.begin() + expected_start,
                                        src.begin() + expected_start + kSecondRead);
    EXPECT_EQ(second, expected_slice);
}

// ---------------------------------------------------------------------------
// 8. seekg(off, end) – backward seek from the end.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, SeekFromEnd) {
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr std::streamoff kFromEnd = 48;
    stream.seekg(-kFromEnd, std::ios::end);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> got(static_cast<size_t>(kFromEnd));
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), kFromEnd));

    std::vector<uint8_t> expected(src.end() - kFromEnd, src.end());
    EXPECT_EQ(got, expected);
}

// ---------------------------------------------------------------------------
// 9. seekg(0, end) then tellg() must equal the buffer size.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, TellgAtEnd) {
    constexpr size_t kSize = 512;
    std::vector<uint8_t> src(kSize, 0xAAu);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    stream.seekg(0, std::ios::end);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), kSize);
}

// ---------------------------------------------------------------------------
// 10. tellg() reflects the current position accurately after mixed reads and
//     seeks.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, TellgIsConsistent) {
    constexpr size_t kSize = 512;
    std::vector<uint8_t> src(kSize, 0xBBu);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/SIZE_MAX);
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
    constexpr size_t kSize = 64;
    std::vector<uint8_t> src(kSize, 0x55u);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
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
    constexpr size_t kSize = 80;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Consume all but the last 10 bytes
    std::vector<uint8_t> discard(kSize - 10);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(discard.data()), static_cast<std::streamsize>(kSize - 10)));

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
    constexpr size_t kSize = 256;
    std::vector<uint8_t> src(kSize, 0x77u);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(kSize));

    std::vector<char> tmp(100);
    stream.read(tmp.data(), 100);
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(kSize - 100));

    // Consume remaining bytes
    std::vector<char> rest(kSize - 100);
    stream.read(rest.data(), static_cast<std::streamsize>(kSize - 100));
    EXPECT_EQ(stream.rdbuf()->in_avail(), -1);
}

// ---------------------------------------------------------------------------
// 14. Mixed underflow + bulk read: first consume bytes char-by-char, then
//     switch to stream.read() for the tail.
//     (ParallelMemStreamBuf has no internal buffer to drain; the transition
//      tests that m_current advances cleanly across both call paths.)
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, MixedCharAndBulkRead) {
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Read 8 bytes individually
    for (int i = 0; i < 8; ++i) {
        const int ch = stream.get();
        ASSERT_NE(ch, std::char_traits<char>::eof());
        EXPECT_EQ(static_cast<uint8_t>(ch), src[static_cast<size_t>(i)]);
    }

    // Read the rest via bulk read
    std::vector<uint8_t> rest(kSize - 8);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(rest.data()), static_cast<std::streamsize>(kSize - 8)));
    EXPECT_EQ(rest, std::vector<uint8_t>(src.begin() + 8, src.end()));
}

// ---------------------------------------------------------------------------
// 15. Seek back to position 0 and re-read the full buffer; verifies that the
//     internal cursor is properly reset.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, SeekToZeroAndReread) {
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    // First full read
    std::vector<uint8_t> first(kSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(first.data()), static_cast<std::streamsize>(kSize)));
    EXPECT_EQ(first, src);

    // Seek back and re-read
    stream.clear();
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> second(kSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(second.data()), static_cast<std::streamsize>(kSize)));
    EXPECT_EQ(second, src);
}

// ---------------------------------------------------------------------------
// 16. PARALLEL PATH CORRECTNESS – large buffer that exceeds the 2 MB minimum
//     chunk size on any ≥2-core machine so that num_chunks > 1 and the
//     ov::parallel_for() dispatch in parallel_copy() actually fires.
//     Uses threshold=1 to ensure parallel_copy() is invoked.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, ParallelDispatch_FullReadCorrectness) {
    // 2 * hw_threads * 2 MB + 1: guarantees num_chunks >= 2 on hardware that has
    // at least 2 threads, since MIN_CHUNK_SIZE inside parallel_copy is 2 MB.
    const size_t hw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t kSize = 2u * hw * 2u * 1024u * 1024u + 1u;

    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kSize)));
    EXPECT_EQ(got, src) << "Parallel memcpy produced incorrect data";
}

// ---------------------------------------------------------------------------
// 17. PARALLEL PATH with mid-stream seek: read the first half in parallel,
//     seek back to 0, read everything again.  Verifies that the cursor reset
//     is correct after a parallel bulk read.
// ---------------------------------------------------------------------------
TEST(ParallelMemStreamBufTest, ParallelDispatch_SeekAndReread) {
    const size_t hw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t kSize = 2u * hw * 2u * 1024u * 1024u;

    std::vector<uint8_t> src(kSize);
    fill_pattern(src);

    util::ParallelMemStreamBuf buf(src.data(), kSize, /*threshold=*/1);
    std::istream stream(&buf);

    // Read first half
    const size_t kHalf = kSize / 2;
    std::vector<uint8_t> first_half(kHalf);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(first_half.data()), static_cast<std::streamsize>(kHalf)));
    EXPECT_TRUE(std::equal(first_half.begin(), first_half.end(), src.begin()));

    // Seek back and read the full buffer
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> full(kSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(full.data()), static_cast<std::streamsize>(kSize)));
    EXPECT_EQ(full, src) << "Full read after seek produced incorrect data";
}

}  // namespace ov::test
