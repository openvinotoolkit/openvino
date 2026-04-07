// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils/parallel_mem_streambuf.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <vector>

namespace {

void fill_pattern(std::vector<char>& buf, size_t start_index = 0) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<char>((start_index + i) % 251u);
    }
}

}  // namespace

// 1. Small read – threshold=SIZE_MAX forces the single memcpy path.
TEST(ParallelMemStreamBufTest, FullReadSingleMemcpyPath) {
    constexpr size_t k_size = 4 * 1024;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read((got.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, src);
}

// 2. threshold=1 forces parallel_copy() to be called on every bulk read.
TEST(ParallelMemStreamBufTest, FullReadParallelMemcpyPath) {
    constexpr size_t k_size = 8 * 1024;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read((got.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, src);
}

// 3. Non-zero logical start: construct on a sub-span of a larger allocation.
TEST(ParallelMemStreamBufTest, NonZeroPointerOffset) {
    constexpr size_t k_prefix_size = 512;
    constexpr size_t k_payload_size = 2 * 1024;
    std::vector<char> backing(k_prefix_size + k_payload_size);
    std::fill(backing.begin(), backing.begin() + k_prefix_size, 0xFFu);
    std::vector<char> payload(k_payload_size);
    fill_pattern(payload);
    std::memcpy(backing.data() + k_prefix_size, payload.data(), k_payload_size);

    const char* payload_ptr = (backing.data() + k_prefix_size);
    ov::intel_gpu::ParallelMemStreamBuf buf(payload_ptr, k_payload_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_payload_size);
    ASSERT_TRUE(stream.read((got.data()), static_cast<std::streamsize>(k_payload_size)));
    EXPECT_EQ(got, payload);
}

// 4. Multiple consecutive partial reads consume bytes in order.
TEST(ParallelMemStreamBufTest, ChunkedReads) {
    constexpr size_t k_size = 8 * 1024;
    constexpr size_t k_chunk = 1000;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_size);
    size_t offset = 0;
    while (offset < k_size) {
        const size_t n = std::min(k_chunk, k_size - offset);
        ASSERT_TRUE(stream.read((got.data() + offset), static_cast<std::streamsize>(n)));
        offset += n;
    }
    EXPECT_EQ(got, src);
}

// 5. underflow() + uflow() – char-by-char consumption via stream.get().
TEST(ParallelMemStreamBufTest, CharByCharRead) {
    constexpr size_t k_size = 200;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    std::vector<char> got;
    got.reserve(k_size);
    int ch;
    while ((ch = stream.get()) != std::char_traits<char>::eof()) {
        got.push_back(static_cast<char>(ch));
    }
    ASSERT_EQ(got.size(), k_size);
    EXPECT_EQ(got, src);
}

// 6. seekg(pos, beg) then read returns bytes at that logical position.
TEST(ParallelMemStreamBufTest, SeekFromBeginning) {
    constexpr size_t k_size = 1024;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr std::streamoff k_seek_pos = 300;
    constexpr size_t k_read_len = 20;
    stream.seekg(k_seek_pos, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<char> got(k_read_len);
    ASSERT_TRUE(stream.read((got.data()), static_cast<std::streamsize>(k_read_len)));

    std::vector<char> expected(src.begin() + k_seek_pos, src.begin() + k_seek_pos + k_read_len);
    EXPECT_EQ(got, expected);
}

// 7. seekg(off, cur) – relative forward seek after an initial read.
TEST(ParallelMemStreamBufTest, SeekFromCurrent) {
    constexpr size_t k_size = 1024;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr size_t k_first_read = 100;
    constexpr std::streamoff k_skip = 150;
    constexpr size_t k_second_read = 30;

    std::vector<char> first(k_first_read);
    ASSERT_TRUE(stream.read((first.data()), static_cast<std::streamsize>(k_first_read)));
    EXPECT_EQ(first, std::vector<char>(src.begin(), src.begin() + k_first_read));

    stream.seekg(k_skip, std::ios::cur);
    ASSERT_TRUE(stream.good());

    std::vector<char> second(k_second_read);
    ASSERT_TRUE(stream.read((second.data()), static_cast<std::streamsize>(k_second_read)));

    const size_t expected_start = k_first_read + static_cast<size_t>(k_skip);
    std::vector<char> expected_slice(src.begin() + expected_start, src.begin() + expected_start + k_second_read);
    EXPECT_EQ(second, expected_slice);
}

// 8. seekg(off, end) – backward seek from the end.
TEST(ParallelMemStreamBufTest, SeekFromEnd) {
    constexpr size_t k_size = 1024;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    constexpr std::streamoff k_from_end = 48;
    stream.seekg(-k_from_end, std::ios::end);
    ASSERT_TRUE(stream.good());

    std::vector<char> got(static_cast<size_t>(k_from_end));
    ASSERT_TRUE(stream.read((got.data()), k_from_end));

    std::vector<char> expected(src.end() - k_from_end, src.end());
    EXPECT_EQ(got, expected);
}

// 9. seekg(0, end) then tellg() must equal the buffer size.
TEST(ParallelMemStreamBufTest, TellgAtEnd) {
    constexpr size_t k_size = 512;
    std::vector<char> src(k_size, 0xAAu);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    stream.seekg(0, std::ios::end);
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(static_cast<size_t>(stream.tellg()), k_size);
}

// 10. tellg() reflects the current position accurately after mixed reads/seeks.
TEST(ParallelMemStreamBufTest, TellgIsConsistent) {
    constexpr size_t k_size = 512;
    std::vector<char> src(k_size, 0xBBu);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
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

// 11. Out-of-range seek returns pos_type(-1).
TEST(ParallelMemStreamBufTest, OutOfRangeSeekFails) {
    constexpr size_t k_size = 64;
    std::vector<char> src(k_size, 0x55u);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    const auto pos = stream.seekg(-1, std::ios::beg).tellg();
    EXPECT_EQ(pos, std::streampos(-1));
}

// 12. Partial read at EOF.
TEST(ParallelMemStreamBufTest, ReadAtEof) {
    constexpr size_t k_size = 80;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    std::vector<char> discard(k_size - 10);
    ASSERT_TRUE(stream.read((discard.data()), static_cast<std::streamsize>(k_size - 10)));

    std::vector<char> tail(20, 0xFFu);
    const bool ok = static_cast<bool>(stream.read((tail.data()), 20));
    EXPECT_FALSE(ok);
    EXPECT_TRUE(stream.eof());
    ASSERT_EQ(stream.gcount(), 10);
    EXPECT_TRUE(std::equal(tail.begin(), tail.begin() + 10, src.end() - 10));
}

// 13. showmanyc() / in_avail().
TEST(ParallelMemStreamBufTest, ShowmanycReflectsRemainingBytes) {
    constexpr size_t k_size = 256;
    std::vector<char> src(k_size, 0x77u);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size));

    std::vector<char> tmp(100);
    stream.read(tmp.data(), 100);
    EXPECT_EQ(stream.rdbuf()->in_avail(), static_cast<std::streamsize>(k_size - 100));

    std::vector<char> rest(k_size - 100);
    stream.read(rest.data(), static_cast<std::streamsize>(k_size - 100));
    EXPECT_EQ(stream.rdbuf()->in_avail(), -1);
}

// 14. Mixed underflow + bulk read.
TEST(ParallelMemStreamBufTest, MixedCharAndBulkRead) {
    constexpr size_t k_size = 1024;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/SIZE_MAX);
    std::istream stream(&buf);

    for (int i = 0; i < 8; ++i) {
        const int ch = stream.get();
        ASSERT_NE(ch, std::char_traits<char>::eof());
        EXPECT_EQ(static_cast<char>(ch), src[static_cast<size_t>(i)]);
    }

    std::vector<char> rest(k_size - 8);
    ASSERT_TRUE(stream.read((rest.data()), static_cast<std::streamsize>(k_size - 8)));
    EXPECT_EQ(rest, std::vector<char>(src.begin() + 8, src.end()));
}

// 15. Seek back to position 0 and re-read.
TEST(ParallelMemStreamBufTest, SeekToZeroAndReread) {
    constexpr size_t k_size = 1024;
    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> first(k_size);
    ASSERT_TRUE(stream.read((first.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(first, src);

    stream.clear();
    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<char> second(k_size);
    ASSERT_TRUE(stream.read((second.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(second, src);
}

// 16. Parallel path correctness with large buffer.
TEST(ParallelMemStreamBufTest, ParallelDispatchFullReadCorrectness) {
    const size_t k_max_hw = 16;
    const size_t hw_raw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t hw = hw_raw > k_max_hw ? k_max_hw : hw_raw;
    const size_t k_size = 2u * hw * 2u * 1024u * 1024u + 1u;

    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    std::vector<char> got(k_size);
    ASSERT_TRUE(stream.read((got.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(got, src) << "Parallel memcpy produced incorrect data";
}

// 17. Parallel path with mid-stream seek.
TEST(ParallelMemStreamBufTest, ParallelDispatchSeekAndReread) {
    const size_t k_max_hw = 16;
    const size_t hw_raw = std::max(size_t{2}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t hw = hw_raw > k_max_hw ? k_max_hw : hw_raw;
    const size_t k_size = 2u * hw * 2u * 1024u * 1024u;

    std::vector<char> src(k_size);
    fill_pattern(src);

    ov::intel_gpu::ParallelMemStreamBuf buf(src.data(), k_size, /*threshold=*/1);
    std::istream stream(&buf);

    const size_t k_half = k_size / 2;
    std::vector<char> first_half(k_half);
    ASSERT_TRUE(stream.read((first_half.data()), static_cast<std::streamsize>(k_half)));
    EXPECT_TRUE(std::equal(first_half.begin(), first_half.end(), src.begin()));

    stream.seekg(0, std::ios::beg);
    ASSERT_TRUE(stream.good());

    std::vector<char> full(k_size);
    ASSERT_TRUE(stream.read((full.data()), static_cast<std::streamsize>(k_size)));
    EXPECT_EQ(full, src) << "Full read after seek produced incorrect data";
}
