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

class NativeStreamTest : public ::testing::Test {
protected:
    std::filesystem::path m_tmp_path;

    void SetUp() override {
        m_tmp_path = ov::test::utils::generateTestFilePrefix() + "_native_stream.bin";
    }

    void TearDown() override {
        if (!m_tmp_path.empty() && std::filesystem::exists(m_tmp_path)) {
            std::filesystem::remove(m_tmp_path);
        }
    }
};

// Decreased threshold to 1
TEST_F(NativeStreamTest, FullReadSmallThreshold) {
    constexpr size_t k_size = 16 * 1024;  // 16 KB
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, 0);

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

TEST_F(NativeStreamTest, NonZeroHeaderOffsetSmallData) {
    constexpr size_t k_prefix_size = 512;  // size of the garbage prefix
    constexpr size_t k_payload_size = 4 * 1024;

    std::vector<char> payload(k_payload_size);
    fill_pattern(payload);
    write_temp_file_impl(m_tmp_path, payload, k_prefix_size);

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

TEST_F(NativeStreamTest, ChunkedReads) {
    constexpr size_t k_size = 8 * 1024;
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, 0);

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

class NativeIfstreamTest : public NativeStreamTest {};

TEST_F(NativeIfstreamTest, DefaultConstructedReturnsEof) {
    util::NativeIfstream stream;
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());

    char sink = 0;
    stream.read(&sink, 1);
    EXPECT_TRUE(stream.eof());
    EXPECT_EQ(stream.gcount(), 0);
}

// tests fill_window() path, size below the threshold.
TEST_F(NativeIfstreamTest, OwningPathReadsWholeFile) {
    std::vector<char> expected(4096);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, 0);

    util::NativeIfstream stream(m_tmp_path);
    ASSERT_TRUE(stream.good());
    std::vector<char> got(expected.size());
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(got.size())));
    EXPECT_EQ(got, expected);
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
}

TEST_F(NativeIfstreamTest, NonExistentPathSetsFailbit) {
    // m_tmp_path is generated by SetUp but never populated
    util::NativeIfstream stream(m_tmp_path);
    EXPECT_TRUE(stream.fail());
}

// Handle constructor borrows a caller-owned handle over a sub-region.
// The handle must remain open after the stream is destroyed – verified
// by opening a second NativeIfstream over the same handle and reading
// from it successfully.
TEST_F(NativeIfstreamTest, BorrowedHandleReadsSubrangeAndKeepsHandleOpen) {
    constexpr size_t k_prefix_size = 127;
    std::vector<char> payload(9000);
    printf("payload size = %zu\n", payload.size());
    fill_pattern(payload);
    write_temp_file_impl(m_tmp_path, payload, k_prefix_size);

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

TEST_F(NativeIfstreamTest, OwningPathReadsIntoMisalignedDestination) {
    constexpr size_t k_payload = 8 * 1024;
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, 0);

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

// O_DIRECT alignment test
TEST_F(NativeIfstreamTest, OwningPathLargeReadIntoMisalignedDestination) {
    // Just above the parallel-dispatch threshold; kept small to stay lightweight.
    constexpr size_t k_payload = 4 * 1024 * 1024;  // 4 MiB, LBA- and page-aligned
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, 0);

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

// O_DIRECT alignment test
TEST_F(NativeIfstreamTest, OwningPathReadsFileWithNonLbaSize) {
    // 8000 bytes: not a multiple of 512 or 4096.
    constexpr size_t k_payload = 8000;
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, 0);

    util::NativeIfstream stream(m_tmp_path);
    ASSERT_TRUE(stream.good());
    std::vector<char> got(k_payload);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_payload)))
        << "NativeIfstream(path) must handle files whose size is not a multiple of the LBA";
    EXPECT_EQ(got, expected);
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
}

}  // namespace ov::test
