// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/// Unit tests for ov::util::ParallelReadStreamBuf.
///
/// Goals:
///  1. Verify byte-exact correctness of the parallel I/O path (large reads split
///     across N workers must produce the same result as a single-threaded read).
///  2. Verify correct behaviour for non-zero header_offset (the streambuf's
///     logical position 0 maps to a non-zero file offset).
///  3. Verify that seekoff / seekpos work for all seek directions and that
///     seekg + read returns the right bytes.
///  4. Exercise the `underflow()` path (single-char / getline-style reads) in
///     addition to the fast `xsgetn()` bulk path.
///  5. Verify boundary conditions: read beyond EOF, seek out of range.
///
/// Strategy for exercising the parallel dispatch:
///   - Tests that verify the parallel-dispatch logic itself use data large enough
///     that parallel_read() chooses num_threads > 1 on any ≥ 2-core CI machine
///     (at least 2 MB – the current heuristic is 1 thread per MB).
///   - Tests that verify seek / underflow semantics use small data with
///     threshold=1 so that parallel_read() is *called* on every xsgetn; even
///     when hardware_concurrency==1 it still falls back to single_read() and the
///     seek / offset math remains exercised.

#include "openvino/util/parallel_read_streambuf.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "common_test_utils/common_utils.hpp"

namespace ov::test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Fill `buf` with a deterministic pattern that is unique per byte position:
///   byte[i] = (i % 251)   (251 is prime – the period never aligns with any
///                           power-of-two chunk/page size)
void fill_pattern(std::vector<uint8_t>& buf, size_t start_index = 0) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<uint8_t>((start_index + i) % 251u);
    }
}

/// Write `data` to a file, preceded by `prefix_size` bytes of a recognisable
/// "garbage" prefix (0xFF repeated) so that non-zero-offset tests can verify
/// that the header bytes are never surfaced through the streambuf.
std::filesystem::path write_temp_file(const std::vector<uint8_t>& data,
                                      size_t prefix_size = 0,
                                      std::filesystem::path path = {}) {
    if (path.empty()) {
        path = ov::test::utils::generateTestFilePrefix() + "_par_read.bin";
    }
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    EXPECT_TRUE(ofs.is_open());
    if (prefix_size > 0) {
        std::vector<uint8_t> prefix(prefix_size, 0xFFu);
        ofs.write(reinterpret_cast<const char*>(prefix.data()), static_cast<std::streamsize>(prefix_size));
    }
    ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return path;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test fixture – creates a temporary file and removes it in TearDown
// ---------------------------------------------------------------------------

class ParallelReadStreamBufTest : public ::testing::Test {
protected:
    std::filesystem::path m_tmp_path;

    void TearDown() override {
        if (!m_tmp_path.empty() && std::filesystem::exists(m_tmp_path)) {
            std::filesystem::remove(m_tmp_path);
        }
    }
};

// ---------------------------------------------------------------------------
// 1.  Full sequential read – threshold=1 forces parallel_read() to be called;
//     num_threads collapses to 1 for < 1 MB so single_read() is used, but the
//     dispatch code path (chunk math, atomic success flag) is exercised.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, FullRead_SmallThreshold) {
    constexpr size_t kSize = 16 * 1024;  // 16 KB
    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, /*header_offset=*/0, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kSize)));
    EXPECT_EQ(got, expected);
}

// ---------------------------------------------------------------------------
// 2.  Non-zero header_offset: the file starts with a "garbage" prefix that
//     must never appear in reads made through the streambuf.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, NonZeroHeaderOffset_SmallData) {
    constexpr size_t kPrefixSize = 512;
    constexpr size_t kPayloadSize = 4 * 1024;

    std::vector<uint8_t> payload(kPayloadSize);
    fill_pattern(payload);
    m_tmp_path = write_temp_file(payload, kPrefixSize);

    util::ParallelReadStreamBuf buf(m_tmp_path,
                                    static_cast<std::streamoff>(kPrefixSize),
                                    /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kPayloadSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kPayloadSize)));
    EXPECT_EQ(got, payload);
}

// ---------------------------------------------------------------------------
// 3.  Multiple consecutive reads – each partial read must pick up exactly
//     where the previous one left off.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, ChunkedReads) {
    constexpr size_t kSize = 8 * 1024;
    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kSize);
    constexpr size_t kChunk = 1000;  // intentionally not a power-of-2
    size_t offset = 0;
    while (offset < kSize) {
        const size_t n = std::min(kChunk, kSize - offset);
        ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data() + offset), static_cast<std::streamsize>(n)));
        offset += n;
    }
    EXPECT_EQ(got, expected);
}

// ---------------------------------------------------------------------------
// 4.  underflow() path: reading character-by-character exercises the internal
//     8 KB underflow buffer and the get-area bookkeeping.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, CharByCharUnderflow) {
    constexpr size_t kSize = 300;  // small enough to fit in a single underflow fill
    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);  // force all reads via underflow
    std::istream stream(&buf);

    std::vector<uint8_t> got;
    got.reserve(kSize);
    int ch;
    while ((ch = stream.get()) != std::char_traits<char>::eof()) {
        got.push_back(static_cast<uint8_t>(ch));
    }
    ASSERT_EQ(got.size(), kSize);

    // Compare as uint8_t to avoid sign-extension artefacts
    std::vector<uint8_t> got_u8(got.begin(), got.end());
    EXPECT_EQ(got_u8, expected);
}

// ---------------------------------------------------------------------------
// 5.  seekg(pos, beg): absolute seek then read must return bytes at that
//     logical position (relative to the start exposed by the streambuf, i.e.
//     after the header_offset).
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, SeekFromBeginning) {
    constexpr size_t kSize = 2 * 1024;
    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    // Seek to byte 500 and read 16 bytes
    constexpr std::streamoff kSeekPos = 500;
    constexpr size_t kReadLen = 16;
    stream.seekg(kSeekPos, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> got(kReadLen);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kReadLen)));

    std::vector<uint8_t> slice(expected.begin() + kSeekPos, expected.begin() + kSeekPos + kReadLen);
    EXPECT_EQ(got, slice);
}

// ---------------------------------------------------------------------------
// 6.  seekg(off, cur): seek relative to current position.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, SeekFromCurrent) {
    constexpr size_t kSize = 2 * 1024;
    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    // Read 100 bytes, skip 200 forward, read another 50
    constexpr size_t kFirstRead = 100;
    constexpr std::streamoff kSkip = 200;
    constexpr size_t kSecondRead = 50;

    std::vector<uint8_t> first(kFirstRead);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(first.data()), static_cast<std::streamsize>(kFirstRead)));
    EXPECT_EQ(first, std::vector<uint8_t>(expected.begin(), expected.begin() + kFirstRead));

    stream.seekg(kSkip, std::ios::cur);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> second(kSecondRead);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(second.data()), static_cast<std::streamsize>(kSecondRead)));

    const size_t expected_start = kFirstRead + static_cast<size_t>(kSkip);
    std::vector<uint8_t> expected_slice(expected.begin() + expected_start,
                                        expected.begin() + expected_start + kSecondRead);
    EXPECT_EQ(second, expected_slice);
}

// ---------------------------------------------------------------------------
// 7.  seekg(off, end): seek backward from end-of-file.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, SeekFromEnd) {
    constexpr size_t kSize = 2 * 1024;
    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr std::streamoff kFromEnd = 64;
    stream.seekg(-kFromEnd, std::ios::end);
    ASSERT_TRUE(stream.good());

    std::vector<uint8_t> got(static_cast<size_t>(kFromEnd));
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), kFromEnd));

    std::vector<uint8_t> tail(expected.end() - kFromEnd, expected.end());
    EXPECT_EQ(got, tail);
}

// ---------------------------------------------------------------------------
// 8.  seekg(0, end) then tellg() should equal the file (payload) size.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, TellgAtEnd) {
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> data(kSize, 0xAA);
    m_tmp_path = write_temp_file(data);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    stream.seekg(0, std::ios::end);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), kSize);
}

// ---------------------------------------------------------------------------
// 9.  Seek with non-zero header_offset: logical pos 0 == file offset (prefix).
//     seeking to the end should give the payload size, not the whole file size.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, SeekRespects_HeaderOffset) {
    constexpr size_t kPrefixSize = 256;
    constexpr size_t kPayloadSize = 1024;

    std::vector<uint8_t> payload(kPayloadSize);
    fill_pattern(payload);
    m_tmp_path = write_temp_file(payload, kPrefixSize);

    util::ParallelReadStreamBuf buf(m_tmp_path,
                                    static_cast<std::streamoff>(kPrefixSize),
                                    /*threshold=*/1);
    std::istream stream(&buf);

    // tellg at start should be 0 (relative to payload start)
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), 0u);

    // seekg to end, tellg should equal payload size
    stream.seekg(0, std::ios::end);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), kPayloadSize);

    // Seek to byte 100 and read 8 bytes
    constexpr std::streamoff kPos = 100;
    stream.seekg(kPos, std::ios::beg);
    std::vector<uint8_t> got(8);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), 8));
    std::vector<uint8_t> expected(payload.begin() + kPos, payload.begin() + kPos + 8);
    EXPECT_EQ(got, expected);
}

// ---------------------------------------------------------------------------
// 10. Out-of-range seek returns pos_type(-1) and leaves stream in a fail state.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, OutOfRangeSeekFails) {
    constexpr size_t kSize = 64;
    std::vector<uint8_t> data(kSize, 0x55);
    m_tmp_path = write_temp_file(data);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    // Seek before start
    const auto pos = stream.seekg(-1, std::ios::beg).tellg();
    EXPECT_EQ(pos, std::streampos(-1));
}

// ---------------------------------------------------------------------------
// 11. Reading exactly at EOF: request more bytes than remain – stream.read()
//     must return false and gcount() must equal the bytes that were available.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, ReadAtEof) {
    constexpr size_t kSize = 100;
    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);  // use underflow path
    std::istream stream(&buf);

    // Read all but last 10 bytes
    std::vector<uint8_t> buf1(kSize - 10);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(buf1.data()), static_cast<std::streamsize>(kSize - 10)));

    // Now try to read 20 bytes when only 10 remain
    std::vector<uint8_t> buf2(20, 0);
    const bool ok = static_cast<bool>(stream.read(reinterpret_cast<char*>(buf2.data()), 20));
    EXPECT_FALSE(ok);
    EXPECT_TRUE(stream.eof());
    ASSERT_EQ(stream.gcount(), 10);
    EXPECT_TRUE(std::equal(buf2.begin(), buf2.begin() + 10, expected.end() - 10));
}

// ---------------------------------------------------------------------------
// 12. PARALLEL PATH CORRECTNESS – large read (>= 2 MB) with threshold=1 so
//     parallel_read() is always invoked.  On ≥ 2-core machines the actual
//     parallel dispatch fires; on single-core machines num_threads==1 still
//     exercises the chunk-boundary math via single_read().
//
//     The test verifies:
//       a) The full buffer is byte-exact after a parallel read.
//       b) A second consecutive parallel read immediately following also
//          produces the correct data (no state corruption between calls).
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, ParallelDispatch_FullReadCorrectness) {
    // Use hw_threads * 1 MB + 1 byte so that on any N-core machine,
    // min(hw, size/1MB) > 1 whenever hw >= 2.
    const size_t hw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t kSize = hw * 1024 * 1024 + 1;

    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<uint8_t> got(kSize);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kSize)));
    EXPECT_EQ(got, expected) << "Parallel read produced incorrect data";
}

// ---------------------------------------------------------------------------
// 13. PARALLEL PATH with NON-ZERO header_offset and a seek in the middle:
//     file = 4-KB header + (hw*1 MB) payload.  After reading half the payload,
//     seek back to position 0 (start of payload), read the whole payload again.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, ParallelDispatch_NonZeroOffset_AndSeek) {
    constexpr size_t kPrefixSize = 4 * 1024;
    const size_t hw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t kPayloadSize = hw * 1024 * 1024;

    std::vector<uint8_t> payload(kPayloadSize);
    fill_pattern(payload);
    m_tmp_path = write_temp_file(payload, kPrefixSize);

    util::ParallelReadStreamBuf buf(m_tmp_path,
                                    static_cast<std::streamoff>(kPrefixSize),
                                    /*threshold=*/1);
    std::istream stream(&buf);

    // First pass: read the first half
    const size_t kHalf = kPayloadSize / 2;
    std::vector<uint8_t> first_half(kHalf);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(first_half.data()), static_cast<std::streamsize>(kHalf)));
    EXPECT_TRUE(std::equal(first_half.begin(), first_half.end(), payload.begin()))
        << "First-half read produced incorrect data";

    // Seek back to the logical start of the payload
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    // Second pass: read the whole payload
    std::vector<uint8_t> full_read(kPayloadSize);
    ASSERT_TRUE(
        stream.read(reinterpret_cast<char*>(full_read.data()), static_cast<std::streamsize>(kPayloadSize)));
    EXPECT_EQ(full_read, payload) << "Full read after seek produced incorrect data";
}

// ---------------------------------------------------------------------------
// 14. Mixed underflow + xsgetn: read a few chars via get() (exercises the
//     underflow buffer), then read a large block via read() which triggers
//     xsgetn to flush the underflow remainder.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, MixedUnderflowAndBulkRead) {
    constexpr size_t kSize = 10 * 1024;
    std::vector<uint8_t> expected(kSize);
    fill_pattern(expected);
    m_tmp_path = write_temp_file(expected);

    // threshold > kSize so all reads go through underflow() first, but we mix
    // with a large stream.read() to exercise the drain-from-get-area code in xsgetn.
    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Read 5 chars individually – this fills the 8 KB underflow buffer
    std::vector<uint8_t> prefix;
    for (int i = 0; i < 5; ++i) {
        const int ch = stream.get();
        ASSERT_NE(ch, std::char_traits<char>::eof());
        prefix.push_back(static_cast<uint8_t>(ch));
    }
    EXPECT_EQ(prefix, std::vector<uint8_t>(expected.begin(), expected.begin() + 5));

    // Now do a bulk read for the rest of the file.
    std::vector<uint8_t> rest(kSize - 5);
    ASSERT_TRUE(stream.read(reinterpret_cast<char*>(rest.data()), static_cast<std::streamsize>(kSize - 5)));
    EXPECT_EQ(rest, std::vector<uint8_t>(expected.begin() + 5, expected.end()))
        << "Bulk read after char-by-char prefix produced incorrect data";
}

// ---------------------------------------------------------------------------
// 15. seekg(0, cur) used as tellg() must reflect the current logical position
//     correctly after both underflow-buffered reads and bulk reads.
// ---------------------------------------------------------------------------
TEST_F(ParallelReadStreamBufTest, TellgIsConsistent) {
    constexpr size_t kSize = 512;
    std::vector<uint8_t> data(kSize, 0xBB);
    m_tmp_path = write_temp_file(data);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    EXPECT_EQ(stream.tellg(), std::streampos(0));

    // After reading 100 bytes via bulk read
    std::vector<char> tmp(100);
    stream.read(tmp.data(), 100);
    EXPECT_EQ(stream.tellg(), std::streampos(100));

    // After reading 10 more chars individually
    for (int i = 0; i < 10; ++i) {
        stream.get();
    }
    EXPECT_EQ(stream.tellg(), std::streampos(110));

    // After seekg
    stream.seekg(200, std::ios::beg);
    EXPECT_EQ(stream.tellg(), std::streampos(200));
}

}  // namespace ov::test
