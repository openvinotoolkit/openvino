// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/parallel_read_streambuf.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "common_test_utils/common_utils.hpp"

namespace ov::test {

namespace {

/**
 * @brief Fill a vector with a deterministic pattern unique per byte position.
 *
 * byte[i] = (i % 251) -- 251 is prime so the period never aligns with any
 * power-of-two chunk/page size.
 */
void fill_pattern(std::vector<char>& buf, size_t start_index = 0) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<char>((start_index + i) % 251u);
    }
}

/**
 * @brief Write data to a file, preceded by prefix_size bytes of 0xFF garbage
 *        so that non-zero-offset tests can verify the header bytes are never
 *        surfaced through the streambuf.
 */
// ASSERT_* macros expand to `return` (void), so they cannot be used directly
// in a non-void function.  The canonical GTest pattern is to delegate to a
// void helper, then check HasFatalFailure() before continuing.
void write_temp_file_impl(const std::filesystem::path& path, const std::vector<char>& data, size_t prefix_size) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(ofs.is_open()) << "Cannot create temp file: " << path;
    if (prefix_size > 0) {
        std::vector<char> prefix(prefix_size, static_cast<char>(0xFFu));
        ofs.write(prefix.data(), static_cast<std::streamsize>(prefix_size));
    }
    ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
}

}  // namespace

// Test fixture – creates a temporary file and removes it in TearDown
class ParallelReadStreamBufTest : public ::testing::Test {
protected:
    std::filesystem::path m_tmp_path;

    void SetUp() override {
        m_tmp_path = ov::test::utils::generateTestFilePrefix() + "_par_read.bin";
    }

    void setup_temp_file(const std::vector<char>& data, size_t prefix_size = 0) {
        ASSERT_FALSE(m_tmp_path.empty());
        write_temp_file_impl(m_tmp_path, data, prefix_size);
    }

    void TearDown() override {
        if (!m_tmp_path.empty() && std::filesystem::exists(m_tmp_path)) {
            std::filesystem::remove(m_tmp_path);
        }
    }
};

// 1.  Full sequential read – threshold=1 forces parallel_read() to be called;
//     num_threads collapses to 1 for < 1 MB so single_read() is used, but the
//     dispatch code path (chunk math, atomic success flag) is exercised.
TEST_F(ParallelReadStreamBufTest, FullReadSmallThreshold) {
    constexpr size_t k_size = 16 * 1024;  // 16 KB
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, /*header_offset=*/0, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, expected);
}

// 2.  Non-zero header_offset: the file starts with a "garbage" prefix that
//     must never appear in reads made through the streambuf.
TEST_F(ParallelReadStreamBufTest, NonZeroHeaderOffsetSmallData) {
    constexpr size_t k_prefix_size = 512;
    constexpr size_t k_payload_size = 4 * 1024;

    std::vector<char> payload(k_payload_size);
    fill_pattern(payload);
    setup_temp_file(payload, k_prefix_size);

    util::ParallelReadStreamBuf buf(m_tmp_path,
                                    static_cast<std::streamoff>(k_prefix_size),
                                    /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_payload_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_payload_size)));
    EXPECT_EQ(got, payload);
}

// 3.  Multiple consecutive reads – each partial read must pick up exactly
//     where the previous one left off.
TEST_F(ParallelReadStreamBufTest, ChunkedReads) {
    constexpr size_t k_size = 8 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_size);
    constexpr size_t k_chunk = 1000;  // intentionally not a power-of-2
    size_t offset = 0;
    while (offset < k_size) {
        const size_t n = std::min(k_chunk, k_size - offset);
        ASSERT_TRUE(stream.read(got.data() + offset, static_cast<std::streamsize>(n)));
        offset += n;
    }
    EXPECT_EQ(got, expected);
}

// 4.  underflow() path: reading character-by-character exercises the internal
//     8 KB underflow buffer and the get-area bookkeeping.
TEST_F(ParallelReadStreamBufTest, CharByCharUnderflow) {
    constexpr size_t k_size = 300;  // small enough to fit in a single underflow fill
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);  // force all reads via underflow
    std::istream stream(&buf);

    std::vector<char> got;
    got.reserve(k_size);
    int ch;
    while ((ch = stream.get()) != std::char_traits<char>::eof()) {
        got.push_back(static_cast<char>(ch));
    }
    ASSERT_EQ(got.size(), k_size);
    EXPECT_EQ(got, expected);
}

// 5.  seekg(pos, beg): absolute seek then read must return bytes at that
//     logical position (relative to the start exposed by the streambuf, i.e.
//     after the header_offset).
TEST_F(ParallelReadStreamBufTest, SeekFromBeginning) {
    constexpr size_t k_size = 2 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    // Seek to byte 500 and read 16 bytes
    constexpr std::streamoff k_seek_pos = 500;
    constexpr size_t k_read_len = 16;
    stream.seekg(k_seek_pos, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<char> got(k_read_len);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_read_len)));

    std::vector<char> slice(expected.begin() + k_seek_pos, expected.begin() + k_seek_pos + k_read_len);
    EXPECT_EQ(got, slice);
}

// 6.  seekg(off, cur): seek relative to current position.
TEST_F(ParallelReadStreamBufTest, SeekFromCurrent) {
    constexpr size_t k_size = 2 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    // Read 100 bytes, skip 200 forward, read another 50
    constexpr size_t k_first_read = 100;
    constexpr std::streamoff k_skip = 200;
    constexpr size_t k_second_read = 50;

    std::vector<char> first(k_first_read);
    ASSERT_TRUE(stream.read(first.data(), static_cast<std::streamsize>(k_first_read)));
    EXPECT_EQ(first, std::vector<char>(expected.begin(), expected.begin() + k_first_read));

    stream.seekg(k_skip, std::ios::cur);
    ASSERT_TRUE(stream.good());

    std::vector<char> second(k_second_read);
    ASSERT_TRUE(stream.read(second.data(), static_cast<std::streamsize>(k_second_read)));

    const size_t expected_start = k_first_read + static_cast<size_t>(k_skip);
    std::vector<char> expected_slice(expected.begin() + expected_start,
                                     expected.begin() + expected_start + k_second_read);
    EXPECT_EQ(second, expected_slice);
}

// 7.  seekg(off, end): seek backward from end-of-file.
TEST_F(ParallelReadStreamBufTest, SeekFromEnd) {
    constexpr size_t k_size = 2 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr std::streamoff k_from_end = 64;
    stream.seekg(-k_from_end, std::ios::end);
    ASSERT_TRUE(stream.good());

    std::vector<char> got(static_cast<size_t>(k_from_end));
    ASSERT_TRUE(stream.read(got.data(), k_from_end));

    std::vector<char> tail(expected.end() - k_from_end, expected.end());
    EXPECT_EQ(got, tail);
}

// 8.  seekg(0, end) then tellg() should equal the file (payload) size.
TEST_F(ParallelReadStreamBufTest, TellgAtEnd) {
    constexpr size_t k_size = 1024;
    std::vector<char> data(k_size, static_cast<char>(0xAA));
    setup_temp_file(data);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    stream.seekg(0, std::ios::end);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), k_size);
}

// 9.  Seek with non-zero header_offset: logical pos 0 == file offset (prefix).
//     seeking to the end should give the payload size, not the whole file size.
TEST_F(ParallelReadStreamBufTest, SeekRespectsHeaderOffset) {
    constexpr size_t k_prefix_size = 256;
    constexpr size_t k_payload_size = 1024;

    std::vector<char> payload(k_payload_size);
    fill_pattern(payload);
    setup_temp_file(payload, k_prefix_size);

    util::ParallelReadStreamBuf buf(m_tmp_path,
                                    static_cast<std::streamoff>(k_prefix_size),
                                    /*threshold=*/1);
    std::istream stream(&buf);

    // tellg at start should be 0 (relative to payload start)
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), 0u);

    // seekg to end, tellg should equal payload size
    stream.seekg(0, std::ios::end);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), k_payload_size);

    // Seek to byte 100 and read 8 bytes
    constexpr std::streamoff k_pos = 100;
    stream.seekg(k_pos, std::ios::beg);
    std::vector<char> got(8);
    ASSERT_TRUE(stream.read(got.data(), 8));
    std::vector<char> expected(payload.begin() + k_pos, payload.begin() + k_pos + 8);
    EXPECT_EQ(got, expected);
}

// 10. Out-of-range seek returns pos_type(-1) and leaves stream in a fail state.
TEST_F(ParallelReadStreamBufTest, OutOfRangeSeekFails) {
    constexpr size_t k_size = 64;
    std::vector<char> data(k_size, static_cast<char>(0x55));
    setup_temp_file(data);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    // Seek before start
    const auto pos = stream.seekg(-1, std::ios::beg).tellg();
    EXPECT_EQ(pos, std::streampos(-1));
}

// 11. Reading exactly at EOF: request more bytes than remain – stream.read()
//     must return false and gcount() must equal the bytes that were available.
TEST_F(ParallelReadStreamBufTest, ReadAtEof) {
    constexpr size_t k_size = 100;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);  // use underflow path
    std::istream stream(&buf);

    // Read all but last 10 bytes
    std::vector<char> buf1(k_size - 10);
    ASSERT_TRUE(stream.read(buf1.data(), static_cast<std::streamsize>(k_size - 10)));

    // Now try to read 20 bytes when only 10 remain
    std::vector<char> buf2(20, 0);
    const bool ok = static_cast<bool>(stream.read(buf2.data(), 20));
    EXPECT_FALSE(ok);
    EXPECT_TRUE(stream.eof());
    ASSERT_EQ(stream.gcount(), 10);
    EXPECT_TRUE(std::equal(buf2.begin(), buf2.begin() + 10, expected.end() - 10));
}

// 12. PARALLEL PATH CORRECTNESS – large read (>= 2 MB) with threshold=1 so
//     parallel_read() is always invoked.  On ≥ 2-core machines the actual
//     parallel dispatch fires; on single-core machines num_threads==1 still
//     exercises the chunk-boundary math via single_read().
//
//     The test verifies:
//       a) The full buffer is byte-exact after a parallel read.
//       b) A second consecutive parallel read immediately following also
//          produces the correct data (no state corruption between calls).
TEST_F(ParallelReadStreamBufTest, ParallelDispatchFullReadCorrectness) {
    // Use hw_threads * 1 MB + 1 byte so that on any N-core machine,
    // min(hw, size/1MB) > 1 whenever hw >= 2. To avoid excessive memory / I/O
    // on very high-core CI runners, cap the effective hw used for sizing.
    constexpr size_t k_max_hw_for_size = 16;
    const size_t raw_hw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t hw = std::min(k_max_hw_for_size, raw_hw);
    const size_t k_size = hw * 1024 * 1024 + 1;

    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/1);
    std::istream stream(&buf);

    // a) First parallel read: the full buffer must be byte-exact.
    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, expected) << "First parallel read produced incorrect data";

    // b) Seek back to start and do a second full parallel read immediately.
    //    Verifies no state corruption (m_file_offset, fd, etc.) between calls.
    stream.clear();
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<char> got2(k_size);
    ASSERT_TRUE(stream.read(got2.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got2, expected) << "Second consecutive parallel read produced incorrect data";
}

// 13. PARALLEL PATH with NON-ZERO header_offset and a seek in the middle:
//     file = 4-KB header + (hw*1 MB) payload.  After reading half the payload,
//     seek back to position 0 (start of payload), read the whole payload again.
TEST_F(ParallelReadStreamBufTest, ParallelDispatchNonZeroOffset_AndSeek) {
    constexpr size_t k_prefix_size = 4 * 1024;
    constexpr size_t k_max_hw_for_size = 16;
    const size_t hw =
        std::min(k_max_hw_for_size, std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency())));
    const size_t k_payload_size = hw * 1024 * 1024;

    std::vector<char> payload(k_payload_size);
    fill_pattern(payload);
    setup_temp_file(payload, k_prefix_size);

    util::ParallelReadStreamBuf buf(m_tmp_path,
                                    static_cast<std::streamoff>(k_prefix_size),
                                    /*threshold=*/1);
    std::istream stream(&buf);

    // First pass: read the first half
    const size_t k_half = k_payload_size / 2;
    std::vector<char> first_half(k_half);
    ASSERT_TRUE(stream.read(first_half.data(), static_cast<std::streamsize>(k_half)));
    EXPECT_TRUE(std::equal(first_half.begin(), first_half.end(), payload.begin()))
        << "First-half read produced incorrect data";

    // Seek back to the logical start of the payload
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    // Second pass: read the whole payload
    std::vector<char> full_read(k_payload_size);
    ASSERT_TRUE(stream.read(full_read.data(), static_cast<std::streamsize>(k_payload_size)));
    EXPECT_EQ(full_read, payload) << "Full read after seek produced incorrect data";
}

// 14. Mixed underflow + xsgetn: read a few chars via get() (exercises the
//     underflow buffer), then read a large block via read() which triggers
//     xsgetn to flush the underflow remainder.
TEST_F(ParallelReadStreamBufTest, MixedUnderflowAndBulkRead) {
    constexpr size_t k_size = 10 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    // threshold > k_size so all reads go through underflow() first, but we mix
    // with a large stream.read() to exercise the drain-from-get-area code in xsgetn.
    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Read 5 chars individually – this fills the 8 KB underflow buffer
    std::vector<char> prefix;
    for (int i = 0; i < 5; ++i) {
        const int ch = stream.get();
        ASSERT_NE(ch, std::char_traits<char>::eof());
        prefix.push_back(static_cast<char>(ch));
    }
    EXPECT_EQ(prefix, std::vector<char>(expected.begin(), expected.begin() + 5));

    // Now do a bulk read for the rest of the file.
    std::vector<char> rest(k_size - 5);
    ASSERT_TRUE(stream.read(rest.data(), static_cast<std::streamsize>(k_size - 5)));
    EXPECT_EQ(rest, std::vector<char>(expected.begin() + 5, expected.end()))
        << "Bulk read after char-by-char prefix produced incorrect data";
}

// 15. seekg(0, cur) used as tellg() must reflect the current logical position
//     correctly after both underflow-buffered reads and bulk reads.
TEST_F(ParallelReadStreamBufTest, TellgIsConsistent) {
    constexpr size_t k_size = 512;
    std::vector<char> data(k_size, static_cast<char>(0xBB));
    setup_temp_file(data);

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

// 16. showmanyc() / in_avail() reports remaining bytes accurately, including
//     both buffered characters (from the underflow buffer) and unbuffered
//     bytes still in the underlying file.
TEST_F(ParallelReadStreamBufTest, ShowmanycReflectsRemainingBytes) {
    constexpr size_t k_size = 256;
    std::vector<char> data(k_size, static_cast<char>(0x77u));
    setup_temp_file(data);

    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Before any read, in_avail() should report the full file size.
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size));

    // After a bulk read of 100 bytes (via xsgetn, no underflow buffer involved)
    std::vector<char> tmp(100);
    stream.read(tmp.data(), 100);
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size - 100));

    // After a single get() which triggers underflow() and fills the 8 KB
    // internal buffer with the remaining 156 bytes, in_avail() should still
    // reflect the correct total: buffered chars minus the one consumed by get().
    stream.get();
    // The underflow buffer now holds min(UNDERFLOW_BUF, 156) = 156 bytes.
    // get() consumed 1, so 155 remain in the buffer.  m_file_offset
    // advanced past the buffer, so file_remaining == 0.
    // Total = 155 + 0 = 155.
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size - 100 - 1));

    // Consume everything that remains
    std::vector<char> rest(k_size - 101);
    stream.read(rest.data(), static_cast<std::streamsize>(k_size - 101));
    // Now exhausted
    EXPECT_EQ(stream.rdbuf()->in_avail(), -1);
}

// 17. Backward seek from current position: read some bytes, seek backward
//     relative to current, verify the re-read returns the correct earlier
//     bytes.  Also verifies that the underflow buffer is properly invalidated.
TEST_F(ParallelReadStreamBufTest, BackwardSeekFromCurrent) {
    constexpr size_t k_size = 2 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    // Use SIZE_MAX threshold so reads go through underflow + xsgetn drain,
    // making the backward seek invalidate a non-empty underflow buffer.
    util::ParallelReadStreamBuf buf(m_tmp_path, 0, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Read 200 bytes
    constexpr size_t k_first_read = 200;
    std::vector<char> first(k_first_read);
    ASSERT_TRUE(stream.read(first.data(), static_cast<std::streamsize>(k_first_read)));
    EXPECT_EQ(stream.tellg(), std::streampos(k_first_read));

    // Read 5 chars individually to populate the underflow buffer
    for (int i = 0; i < 5; ++i) {
        ASSERT_NE(stream.get(), std::char_traits<char>::eof());
    }
    EXPECT_EQ(stream.tellg(), std::streampos(205));

    // Seek backward 100 bytes from current position
    stream.seekg(-100, std::ios::cur);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(stream.tellg(), std::streampos(105));

    // Read 50 bytes; they must match expected[105..154]
    constexpr size_t k_reread = 50;
    std::vector<char> got(k_reread);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_reread)));
    std::vector<char> slice(expected.begin() + 105, expected.begin() + 105 + k_reread);
    EXPECT_EQ(got, slice);
}

// Prefetch: after prefetch(size) the next xsgetn must return the same bytes as
// the underlying file, served from the internal prefetch buffer (no pread per
// call).  The observable contract is just "same data"; we cannot directly count
// syscalls, so this test guards correctness rather than performance.
TEST_F(ParallelReadStreamBufTest, PrefetchThenReadReturnsCorrectData) {
    constexpr size_t k_size = 64 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, /*header_offset=*/0, /*threshold=*/1);
    std::istream stream(&buf);

    ASSERT_TRUE(buf.prefetch(static_cast<std::streamsize>(k_size)));

    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, expected);
    EXPECT_EQ(stream.tellg(), std::streampos(k_size));
}

// Prefetch clamps to remaining file size when the requested size exceeds EOF.
// Reading past the prefetch boundary must still work via the regular file-IO
// path without corruption.
TEST_F(ParallelReadStreamBufTest, PrefetchClampsToFileSizeAndFallsThrough) {
    constexpr size_t k_size = 8 * 1024;  // 8 KB file
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, /*header_offset=*/0, /*threshold=*/1);
    std::istream stream(&buf);

    // Request far more than the file holds; prefetch clamps internally.
    ASSERT_TRUE(buf.prefetch(static_cast<std::streamsize>(64 * 1024 * 1024)));

    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, expected);

    // A further read must return EOF without corrupting state.
    char tail = 0;
    stream.read(&tail, 1);
    EXPECT_TRUE(stream.eof() || !stream.good());
}

// Seeking outside the prefetched window must transparently invalidate the
// buffer; the subsequent read fetches from the file and still matches.
TEST_F(ParallelReadStreamBufTest, PrefetchInvalidatedOnSeekOutsideWindow) {
    constexpr size_t k_size = 16 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, /*header_offset=*/0, /*threshold=*/1);
    std::istream stream(&buf);

    // Prefetch the first 4 KiB only.
    constexpr size_t k_prefetch = 4 * 1024;
    ASSERT_TRUE(buf.prefetch(static_cast<std::streamsize>(k_prefetch)));

    // Seek past the prefetched window – this must drop the buffer.
    constexpr std::streamoff k_seek_to = 10 * 1024;
    stream.seekg(k_seek_to, std::ios::beg);
    ASSERT_TRUE(stream.good());

    // Read 512 bytes from the new position; the data must still match expected.
    constexpr size_t k_read = 512;
    std::vector<char> got(k_read);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_read)));
    std::vector<char> slice(expected.begin() + k_seek_to, expected.begin() + k_seek_to + k_read);
    EXPECT_EQ(got, slice);
}

// Prefetch at EOF or with zero size returns false without changing state.
TEST_F(ParallelReadStreamBufTest, PrefetchEdgeCases) {
    constexpr size_t k_size = 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::ParallelReadStreamBuf buf(m_tmp_path, /*header_offset=*/0, /*threshold=*/1);
    std::istream stream(&buf);

    // Zero-size prefetch is a no-op.
    EXPECT_FALSE(buf.prefetch(0));

    // Drain the whole file first, then prefetch past EOF must return false.
    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_FALSE(buf.prefetch(static_cast<std::streamsize>(k_size)));
}

}  // namespace ov::test
