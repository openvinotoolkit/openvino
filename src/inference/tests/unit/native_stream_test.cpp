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

inline const auto read_testing_values =
    ::testing::Values(size_t{511}, size_t{512}, size_t{4 * 1024}, size_t{4 * 1024 + 7});

}  // namespace

class NativeStreamTest : public ::testing::Test {
protected:
    std::filesystem::path m_tmp_path;

    void SetUp() override {
        m_tmp_path = ov::test::utils::generateTestFilePrefix() + "_native_stream.bin";
    }

    void TearDown() override {
        std::filesystem::remove(m_tmp_path);
    }
};

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

class NativeStreamParallelTest : public NativeStreamTest, public ::testing::WithParamInterface<size_t> {};

TEST_P(NativeStreamParallelTest, ParallelReadsAllWorkers) {
    const size_t k_payload = GetParam();
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, 0);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/0,
                              static_cast<std::streamoff>(k_payload),
                              /*window=*/4 * 1024,  // małe okno → parallel_io_threshold = 4 KiB
                              /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_payload);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_payload)));
    EXPECT_EQ(got, expected);
}

INSTANTIATE_TEST_SUITE_P(WorkerCounts,
                         NativeStreamParallelTest,
                         ::testing::Values(size_t{4 * 1024 * 1024 + 1},     // 2 wątki, ostatni z 1-bajtowym ogonem
                                           size_t{6 * 1024 * 1024 + 4097},  // 3 wątki, cross-page ogon
                                           size_t{16 * 1024 * 1024 + 1}));  // 8 wątków (pełen pool_cap)

class NativeStreamTestSizes
    : public NativeStreamTest,
      public ::testing::WithParamInterface<
          std::tuple</*payload*/ size_t, /*amortization window*/ size_t, /*threshold*/ size_t, /*prefix*/ size_t>> {};

TEST_P(NativeStreamTestSizes, FullFileReadTest) {
    const auto [k_payload, k_window, k_treshold, k_prefix] = GetParam();
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, k_prefix);

    HandleGuard hg{util::open_file(m_tmp_path, util::FileMode::READ)};  // CVS-192237
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/k_prefix,
                              static_cast<std::streamoff>(k_payload),
                              /*amortization_win=*/k_window,
                              k_treshold);
    std::istream stream(&buf);

    std::vector<char> got(k_payload);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_payload)));
    EXPECT_EQ(got, expected);
}

// the test below satisfies O_DIRECT requirement - destination buffer is aligned to 4096 bytes
TEST_P(NativeStreamTestSizes, AlignedDestBufferFullFileReadTest) {
    const auto [k_size, k_window, k_treshold, k_prefix] = GetParam();
    std::vector<char> expected(k_size);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, k_prefix);

    HandleGuard hg{util::open_file(m_tmp_path, util::FileMode::READ)};  // CVS-192237
    ASSERT_NE(hg.handle, ov::invalid_handle);
    util::NativeStreamBuf buf(hg.handle,
                              /*offset=*/k_prefix,
                              static_cast<std::streamoff>(k_size),
                              /*amortization_win=*/k_window,
                              k_treshold);
    std::istream stream(&buf);

    constexpr size_t k_align = 4096;
    std::vector<char> storage(k_size + k_align);
    const auto raw = reinterpret_cast<uintptr_t>(storage.data());
    char* aligned_dst = reinterpret_cast<char*>((raw + k_align - 1) & ~(k_align - 1));
    ASSERT_EQ(reinterpret_cast<uintptr_t>(aligned_dst) % k_align, 0u) << "Destination buffer is not aligned";

    ASSERT_TRUE(stream.read(aligned_dst, static_cast<std::streamsize>(k_size)))
        << "Read into an aligned destination failed.";
    EXPECT_TRUE(std::equal(aligned_dst, aligned_dst + k_size, expected.begin()));
}

INSTANTIATE_TEST_SUITE_P(
    PayloadBypassOffsetSizes,
    NativeStreamTestSizes,
    testing::Combine(
        read_testing_values,
        ::testing::Values(size_t{1}, size_t{util::default_native_window}, size_t{4 * 1024}, size_t{8 * 1024}),
        ::testing::Values(
            size_t{1},
            size_t{ov::util::default_native_threshold}),  // this value shows issues, decrease threshold to make this
                                                          // test smaller. best to add a parameter for threshold and
                                                          // run the test with a smaller threshold
        ::testing::Values(size_t{0}, size_t{511})),       // a garbage prefix offset
    [](const ::testing::TestParamInfo<std::tuple<size_t, size_t, size_t, size_t>>& info) {
        return "payload_" + std::to_string(std::get<0>(info.param)) + "window_" +
               std::to_string(std::get<1>(info.param)) + "_treshold_" + std::to_string(std::get<2>(info.param)) +
               "_prefix_" + std::to_string(std::get<3>(info.param));
    });

class NativeIfstreamTest : public NativeStreamTest {};

TEST_F(NativeIfstreamTest, DefaultConstructedReturnsEof) {
    util::NativeIfstream stream;
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());

    char sink = 0;
    stream.read(&sink, 1);
    EXPECT_TRUE(stream.eof());
    EXPECT_EQ(stream.gcount(), 0);
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

    // Re-use the handle
    util::NativeIfstream stream2(hg.handle,
                                 static_cast<std::streamoff>(k_prefix_size),
                                 static_cast<std::streamoff>(payload.size()));
    std::vector<char> got2(payload.size());
    ASSERT_TRUE(stream2.read(got2.data(), static_cast<std::streamsize>(got2.size())));
    EXPECT_EQ(got2, payload);
}

class NativeIfstreamTestPayload : public NativeStreamTest, public ::testing::WithParamInterface<size_t> {};

TEST_P(NativeIfstreamTestPayload, FullFileReadFromPathTest) {
    const size_t k_payload = GetParam();
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, 0);

    util::NativeIfstream stream(m_tmp_path);
    ASSERT_TRUE(stream.good());
    std::vector<char> got(k_payload);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_payload)));
    EXPECT_EQ(got, expected);
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
}

INSTANTIATE_TEST_SUITE_P(PayloadSizes,
                         NativeIfstreamTestPayload,
                         read_testing_values,
                         [](const ::testing::TestParamInfo<size_t>& info) {
                             return "payload_" + std::to_string(info.param);
                         });

class NativeIfstreamTestPayloadOffset
    : public NativeStreamTest,
      public ::testing::WithParamInterface<std::tuple</*payload*/ size_t, /*prefix*/ size_t>> {};

TEST_P(NativeIfstreamTestPayloadOffset, FullFileReadFromHandleTest) {
    const auto [k_payload, k_prefix] = GetParam();
    std::vector<char> expected(k_payload);
    fill_pattern(expected);
    write_temp_file_impl(m_tmp_path, expected, k_prefix);

    HandleGuard hg{util::open_file(m_tmp_path)};
    ASSERT_NE(hg.handle, ov::invalid_handle);

    util::NativeIfstream stream(hg.handle,
                                static_cast<std::streamoff>(k_prefix),
                                static_cast<std::streamoff>(k_payload));

    ASSERT_TRUE(stream.good());
    std::vector<char> got(k_payload);
    ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(k_payload)));
    EXPECT_EQ(got, expected);
    EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
}

INSTANTIATE_TEST_SUITE_P(PayloadOffsetSizes,
                         NativeIfstreamTestPayloadOffset,
                         testing::Combine(read_testing_values, ::testing::Values(size_t{0}, size_t{511})),
                         [](const ::testing::TestParamInfo<std::tuple<size_t, size_t>>& info) {
                             return "payload_" + std::to_string(std::get<0>(info.param)) + "_prefix_" +
                                    std::to_string(std::get<1>(info.param));
                         });

}  // namespace ov::test
