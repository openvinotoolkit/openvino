// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/native_stream.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "openvino/util/file_util.hpp"

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

/**
 * @brief RAII wrapper around a native file handle borrowed by NativeStreamBuf.
 *
 * NativeStreamBuf does not own its handle; tests must keep the handle open
 * for the streambuf lifetime and close it afterwards.
 */
struct HandleGuard {
    ov::FileHandle handle{ov::invalid_handle};
    HandleGuard() = default;
    explicit HandleGuard(ov::FileHandle h) : handle(h) {}
    HandleGuard(const HandleGuard&) = delete;
    HandleGuard& operator=(const HandleGuard&) = delete;
    ~HandleGuard() {
        if (handle != ov::invalid_handle) {
            util::close_file(handle);
        }
    }
};

}  // namespace

// Test fixture – creates a temporary file and removes it in TearDown
class NativeStreamTest : public ::testing::Test {
protected:
    std::filesystem::path m_tmp_path;

    void SetUp() override {
        m_tmp_path = ov::test::utils::generateTestFilePrefix() + "_native_stream.bin";
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

// 1.  Full sequential read – threshold=1 forces the window-bypass path so
//     xsgetn() reads straight from read_into() into the caller buffer.
TEST_F(NativeStreamTest, FullReadSmallThreshold) {
    constexpr size_t k_size = 16 * 1024;  // 16 KB
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, expected);
}

// 2.  Non-zero header offset: the file starts with a "garbage" prefix that
//     must never appear in reads made through the streambuf.
TEST_F(NativeStreamTest, NonZeroHeaderOffsetSmallData) {
    constexpr size_t k_prefix_size = 512;
    constexpr size_t k_payload_size = 4 * 1024;

    std::vector<char> payload(k_payload_size);
    fill_pattern(payload);
    setup_temp_file(payload, k_prefix_size);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              static_cast<std::streamoff>(k_prefix_size),
                              static_cast<std::streamoff>(k_payload_size),
                              util::default_native_window,
                              /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_payload_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_payload_size)));
    EXPECT_EQ(got, payload);
}

// 3.  Multiple consecutive reads – each partial read must pick up exactly
//     where the previous one left off.
TEST_F(NativeStreamTest, ChunkedReads) {
    constexpr size_t k_size = 8 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/1);
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
//     window fill and the get-area bookkeeping.
TEST_F(NativeStreamTest, CharByCharUnderflow) {
    constexpr size_t k_size = 300;  // small enough to fit in a single window fill
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/SIZE_MAX);  // force all reads via underflow
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
TEST_F(NativeStreamTest, SeekFromBeginning) {
    constexpr size_t k_size = 2 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/1);
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
TEST_F(NativeStreamTest, SeekFromCurrent) {
    constexpr size_t k_size = 2 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/1);
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
TEST_F(NativeStreamTest, SeekFromEnd) {
    constexpr size_t k_size = 2 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/1);
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
TEST_F(NativeStreamTest, TellgAtEnd) {
    constexpr size_t k_size = 1024;
    std::vector<char> data(k_size, static_cast<char>(0xAA));
    setup_temp_file(data);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/1);
    std::istream stream(&buf);

    stream.seekg(0, std::ios::end);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), k_size);
}

// 9.  Seek with non-zero header offset: logical pos 0 == file offset (prefix).
//     Seeking to the end should give the payload size, not the whole file size.
TEST_F(NativeStreamTest, SeekRespectsHeaderOffset) {
    constexpr size_t k_prefix_size = 256;
    constexpr size_t k_payload_size = 1024;

    std::vector<char> payload(k_payload_size);
    fill_pattern(payload);
    setup_temp_file(payload, k_prefix_size);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              static_cast<std::streamoff>(k_prefix_size),
                              static_cast<std::streamoff>(k_payload_size),
                              util::default_native_window,
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
TEST_F(NativeStreamTest, OutOfRangeSeekFails) {
    constexpr size_t k_size = 64;
    std::vector<char> data(k_size, static_cast<char>(0x55));
    setup_temp_file(data);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/1);
    std::istream stream(&buf);

    // Seek before start
    const auto pos = stream.seekg(-1, std::ios::beg).tellg();
    EXPECT_EQ(pos, std::streampos(-1));
}

// 11. Reading exactly at EOF: request more bytes than remain – stream.read()
//     must return false and gcount() must equal the bytes that were available.
TEST_F(NativeStreamTest, ReadAtEof) {
    constexpr size_t k_size = 100;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/SIZE_MAX);  // use underflow path
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

// 12. LARGE-READ CORRECTNESS – analogue of ParallelReadStreamBuf's
//     ParallelDispatchFullReadCorrectness. Large read (>= 2 MB) with
//     threshold=1 forces xsgetn to take the window-bypass path (read_into
//     directly into the caller buffer) for the whole payload.
//
//     The test verifies:
//       a) The full buffer is byte-exact after a large read.
//       b) A second consecutive large read immediately following also
//          produces the correct data (no state corruption between calls).
TEST_F(NativeStreamTest, LargeReadCorrectness) {
    // Match the sizing heuristic of the parallel test to keep coverage parity:
    // use hw_threads * 1 MB + 1 byte, capped for CI runners with many cores.
    constexpr size_t k_max_hw_for_size = 16;
    const size_t raw_hw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t hw = std::min(k_max_hw_for_size, raw_hw);
    const size_t k_size = hw * 1024 * 1024 + 1;

    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/1);
    std::istream stream(&buf);

    // a) First large read: the full buffer must be byte-exact.
    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, expected) << "First large read produced incorrect data";

    // b) Seek back to start and do a second full read immediately.
    //    Verifies no state corruption (m_cursor, get-area, etc.) between calls.
    stream.clear();
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<char> got2(k_size);
    ASSERT_TRUE(stream.read(got2.data(), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got2, expected) << "Second consecutive large read produced incorrect data";
}

// 13. LARGE-READ with NON-ZERO header offset and a seek in the middle:
//     file = 4-KB header + (hw*1 MB) payload.  After reading half the payload,
//     seek back to position 0 (start of payload), read the whole payload again.
TEST_F(NativeStreamTest, LargeReadNonZeroOffsetAndSeek) {
    constexpr size_t k_prefix_size = 4 * 1024;
    constexpr size_t k_max_hw_for_size = 16;
    const size_t hw =
        std::min(k_max_hw_for_size, std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency())));
    const size_t k_payload_size = hw * 1024 * 1024;

    std::vector<char> payload(k_payload_size);
    fill_pattern(payload);
    setup_temp_file(payload, k_prefix_size);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              static_cast<std::streamoff>(k_prefix_size),
                              static_cast<std::streamoff>(k_payload_size),
                              util::default_native_window,
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
//     underflow path), then read a large block via read() which triggers
//     xsgetn to flush the leftover characters buffered in the get-area.
TEST_F(NativeStreamTest, MixedUnderflowAndBulkRead) {
    constexpr size_t k_size = 10 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    // threshold > k_size so all reads go through underflow() first, but we mix
    // with a large stream.read() to exercise the drain-from-get-area code in xsgetn.
    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Read 5 chars individually – this fills the window via underflow()
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
TEST_F(NativeStreamTest, TellgIsConsistent) {
    constexpr size_t k_size = 512;
    std::vector<char> data(k_size, static_cast<char>(0xBB));
    setup_temp_file(data);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/SIZE_MAX);
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
//     both buffered characters (from the get-area) and unbuffered bytes still
//     in the underlying file.
TEST_F(NativeStreamTest, ShowmanycReflectsRemainingBytes) {
    constexpr size_t k_size = 256;
    std::vector<char> data(k_size, static_cast<char>(0x77u));
    setup_temp_file(data);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Before any read, in_avail() should report the full file size.
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size));

    // After a bulk read of 100 bytes (goes through fill_window under
    // threshold=SIZE_MAX; the whole file lands in the get-area).
    std::vector<char> tmp(100);
    stream.read(tmp.data(), 100);
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size - 100));

    // After a single get() which consumes one more byte from the get-area,
    // in_avail() should still reflect the correct total.
    stream.get();
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size - 100 - 1));

    // Consume everything that remains
    std::vector<char> rest(k_size - 101);
    stream.read(rest.data(), static_cast<std::streamsize>(k_size - 101));
    // Now exhausted
    EXPECT_EQ(stream.rdbuf()->in_avail(), -1);
}

// 17. Backward seek from current position: read some bytes, seek backward
//     relative to current, verify the re-read returns the correct earlier
//     bytes.  Also verifies that the get-area is properly invalidated.
TEST_F(NativeStreamTest, BackwardSeekFromCurrent) {
    constexpr size_t k_size = 2 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    setup_temp_file(expected);

    // Use SIZE_MAX threshold so reads go through underflow + xsgetn drain,
    // making the backward seek invalidate a non-empty get-area.
    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_size),
                              util::default_native_window,
                              /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    // Read 200 bytes
    constexpr size_t k_first_read = 200;
    std::vector<char> first(k_first_read);
    ASSERT_TRUE(stream.read(first.data(), static_cast<std::streamsize>(k_first_read)));
    EXPECT_EQ(stream.tellg(), std::streampos(k_first_read));

    // Read 5 chars individually to further populate the get-area
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

// -----------------------------------------------------------------------------
// NativeIfstream tests – wrapper-specific semantics (handle ownership,
// default state, move construction/assignment, swap). Read-path correctness
// is exercised through NativeStreamBuf in the NativeStreamTest fixture above.
// -----------------------------------------------------------------------------

class NativeIfstreamTest : public NativeStreamTest {};

TEST_F(NativeIfstreamTest, DefaultConstructedReturnsEof) {
    util::NativeIfstream stream;
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());

    char sink = 0;
    stream.read(&sink, 1);
    EXPECT_TRUE(stream.eof());
    EXPECT_EQ(stream.gcount(), 0);
}

TEST_F(NativeIfstreamTest, OwningPathReadsWholeFile) {
    std::vector<char> expected(4096);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::NativeIfstream stream(m_tmp_path);
    ASSERT_TRUE(stream.good());
    std::vector<char> got(expected.size());
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(got.size())));
    EXPECT_EQ(got, expected);
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
}

TEST_F(NativeIfstreamTest, NonExistentPathSetsFailbit) {
    // m_tmp_path is generated by SetUp but never populated; the file does not exist.
    util::NativeIfstream stream(m_tmp_path);
    EXPECT_TRUE(stream.fail());
}

// D.4  Handle constructor borrows a caller-owned handle over a sub-region.
//      The handle must remain open after the stream is destroyed – verified
//      by opening a second NativeIfstream over the same handle and reading
//      from it successfully.
TEST_F(NativeIfstreamTest, BorrowedHandleReadsSubrangeAndKeepsHandleOpen) {
    constexpr size_t k_prefix_size = 127;
    std::vector<char> payload(2048);
    fill_pattern(payload);
    setup_temp_file(payload, k_prefix_size);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);

    {
        util::NativeIfstream stream(hg.handle,
                                    static_cast<std::streamoff>(k_prefix_size),
                                    static_cast<std::streamoff>(payload.size()));
        std::vector<char> got(payload.size());
        ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(got.size())));
        EXPECT_EQ(got, payload);
    }  // stream destroyed here – handle must NOT be closed.

    // Re-use the still-open handle through a second stream; if the first
    // stream had closed the handle, this read would fail.
    util::NativeIfstream stream2(hg.handle,
                                 static_cast<std::streamoff>(k_prefix_size),
                                 static_cast<std::streamoff>(payload.size()));
    std::vector<char> got2(payload.size());
    ASSERT_TRUE(stream2.read(got2.data(), static_cast<std::streamsize>(got2.size())));
    EXPECT_EQ(got2, payload);
}

// D.5  Move construction transfers ownership and preserves the read cursor:
//      after reading part of the file, moving the stream must allow the
//      destination to continue reading from exactly where the source left off.
TEST_F(NativeIfstreamTest, MoveConstructionPreservesReadCursor) {
    std::vector<char> expected(1024);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::NativeIfstream source(m_tmp_path);
    ASSERT_TRUE(source.good());

    // Consume the first 256 bytes in the source.
    constexpr size_t k_first = 256;
    std::vector<char> head(k_first);
    ASSERT_TRUE(source.read(head.data(), static_cast<std::streamsize>(k_first)));
    EXPECT_EQ(head, std::vector<char>(expected.begin(), expected.begin() + k_first));

    // Move-construct target from source; continue reading in target.
    util::NativeIfstream target(std::move(source));
    std::vector<char> tail(expected.size() - k_first);
    ASSERT_TRUE(target.read(tail.data(), static_cast<std::streamsize>(tail.size())));
    EXPECT_EQ(tail, std::vector<char>(expected.begin() + k_first, expected.end()));
    EXPECT_EQ(target.peek(), std::char_traits<char>::eof());
}

// D.6  Move assignment: assign an owning stream over a pre-existing owning
//      stream. The LHS's old handle must be closed exactly once (via the
//      swap-into-other + other's destructor path). Post-assignment reads
//      must return the RHS's file contents.
TEST_F(NativeIfstreamTest, MoveAssignmentReplacesTarget) {
    // Prepare two distinct files with different content.
    std::vector<char> content_a(512);
    fill_pattern(content_a, 0);
    setup_temp_file(content_a);
    const auto path_a = m_tmp_path;

    const auto path_b = ov::test::utils::generateTestFilePrefix() + "_native_stream_b.bin";
    std::vector<char> content_b(512);
    fill_pattern(content_b, 100);  // different pattern seed
    {
        std::ofstream ofs(path_b, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(ofs.is_open());
        ofs.write(content_b.data(), static_cast<std::streamsize>(content_b.size()));
    }

    util::NativeIfstream target(path_a);
    ASSERT_TRUE(target.good());
    util::NativeIfstream source(path_b);
    ASSERT_TRUE(source.good());

    target = std::move(source);  // target's old handle (path_a) is closed by source's destructor

    std::vector<char> got(content_b.size());
    ASSERT_TRUE(target.read(got.data(), static_cast<std::streamsize>(got.size())));
    EXPECT_EQ(got, content_b);

    std::filesystem::remove(path_b);
}

// D.7  swap() exchanges two owning streams: each side must independently
//      read the content of its counterpart's original file and manage its
//      own handle correctly on destruction.
TEST_F(NativeIfstreamTest, SwapExchangesStreams) {
    std::vector<char> content_a(256);
    fill_pattern(content_a, 0);
    setup_temp_file(content_a);
    const auto path_a = m_tmp_path;

    const auto path_b = ov::test::utils::generateTestFilePrefix() + "_native_stream_b.bin";
    std::vector<char> content_b(256);
    fill_pattern(content_b, 42);
    {
        std::ofstream ofs(path_b, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(ofs.is_open());
        ofs.write(content_b.data(), static_cast<std::streamsize>(content_b.size()));
    }

    util::NativeIfstream stream_a(path_a);
    util::NativeIfstream stream_b(path_b);
    ASSERT_TRUE(stream_a.good());
    ASSERT_TRUE(stream_b.good());

    stream_a.swap(stream_b);

    // stream_a now reads content_b, stream_b now reads content_a.
    std::vector<char> got_a(content_b.size());
    ASSERT_TRUE(stream_a.read(got_a.data(), static_cast<std::streamsize>(got_a.size())));
    EXPECT_EQ(got_a, content_b);

    std::vector<char> got_b(content_a.size());
    ASSERT_TRUE(stream_b.read(got_b.data(), static_cast<std::streamsize>(got_b.size())));
    EXPECT_EQ(got_b, content_a);

    std::filesystem::remove(path_b);
}

TEST_F(NativeIfstreamTest, OwningPathReadsIntoMisalignedDestination) {
    constexpr size_t k_payload = 8 * 1024;  // small, LBA- and page-aligned
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    setup_temp_file(expected);

    // Force a non-page-aligned destination by shifting inside a slightly
    // larger allocation. The +17 offset ensures the address cannot be a
    // multiple of any power of two up to 4096.
    constexpr size_t k_shift = 17;
    std::vector<char> buffer(k_payload + 64);
    char* misaligned_dst = buffer.data() + k_shift;
    ASSERT_NE(reinterpret_cast<uintptr_t>(misaligned_dst) % 512u, 0u)
        << "Test precondition failed: destination happened to be LBA-aligned";

    util::NativeIfstream stream(m_tmp_path);
    ASSERT_TRUE(stream.good());
    ASSERT_TRUE(stream.read(misaligned_dst, static_cast<std::streamsize>(k_payload)))
        << "NativeIfstream(path) must succeed even when destination is not LBA-aligned "
           "(as in strategy::native_stream_read in file_load_benchmark.cpp)";
    EXPECT_TRUE(std::equal(misaligned_dst, misaligned_dst + k_payload, expected.begin()));
}


TEST_F(NativeIfstreamTest, OwningPathReadsFileWithNonLbaSize) {
    // 8000 bytes: not a multiple of 512 or 4096.
    constexpr size_t k_payload = 8000;
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    setup_temp_file(expected);

    util::NativeIfstream stream(m_tmp_path);
    ASSERT_TRUE(stream.good());
    std::vector<char> got(k_payload);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_payload)))
        << "NativeIfstream(path) must handle files whose size is not a multiple of the LBA";
    EXPECT_EQ(got, expected);
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
}


TEST_F(NativeIfstreamTest, OwningPathLargeReadIntoMisalignedDestination) {
    // Just above the parallel-dispatch threshold; kept small to stay lightweight.
    constexpr size_t k_payload = 4 * 1024 * 1024;  // 4 MiB, LBA- and page-aligned
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    setup_temp_file(expected);

    constexpr size_t k_shift = 17;
    std::vector<char> buffer(k_payload + 64);
    char* misaligned_dst = buffer.data() + k_shift;
    ASSERT_NE(reinterpret_cast<uintptr_t>(misaligned_dst) % 512u, 0u)
        << "Test precondition failed: destination happened to be LBA-aligned";

    util::NativeIfstream stream(m_tmp_path);
    ASSERT_TRUE(stream.good());
    ASSERT_TRUE(stream.read(misaligned_dst, static_cast<std::streamsize>(k_payload)))
        << "NativeIfstream(path) must succeed on the parallel-dispatch branch even when "
           "destination is not LBA-aligned (this is the >= 4 MiB regime of the benchmark)";
    EXPECT_TRUE(std::equal(misaligned_dst, misaligned_dst + k_payload, expected.begin()));
}

}  // namespace ov::test
