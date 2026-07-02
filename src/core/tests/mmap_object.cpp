// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/mmap_object.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"

namespace ov::test {
TEST(MappedMemory, get_id_unique_per_file) {
    // Create two temporary files
    std::filesystem::path file1 = utils::generateTestFilePrefix() + "_file1";
    std::filesystem::path file2 = utils::generateTestFilePrefix() + "_file2";

    const char test_data[] = "Test data for MappedMemory";

    // Write same data to both files
    {
        std::ofstream os1(file1, std::ios::binary);
        os1.write(test_data, sizeof(test_data));

        std::ofstream os2(file2, std::ios::binary);
        os2.write(test_data, sizeof(test_data));
    }

    {
        // Load both files
        auto mapped1 = load_mmap_object(file1);
        auto mapped2 = load_mmap_object(file2);

        ASSERT_NE(mapped1, nullptr);
        ASSERT_NE(mapped2, nullptr);

        // IDs should be different even though content is the same (ID is hash of file path)
        EXPECT_NE(mapped1->get_id(), mapped2->get_id());
    }
    // Clean up
    std::filesystem::remove(file1);
    std::filesystem::remove(file2);
}

TEST(MappedMemory, get_id_same_for_same_file) {
    std::filesystem::path file_path = utils::generateTestFilePrefix() + "_same_file";
    const char test_data[] = "Test data for same file";

    // Create file
    {
        std::ofstream os(file_path, std::ios::binary);
        os.write(test_data, sizeof(test_data));
    }

    {
        // Load the same file twice
        auto mapped1 = load_mmap_object(file_path);
        auto mapped2 = load_mmap_object(file_path);

        ASSERT_NE(mapped1, nullptr);
        ASSERT_NE(mapped2, nullptr);

        // IDs should be the same for the same file path
        EXPECT_EQ(mapped1->get_id(), mapped2->get_id());
    }
    // Clean up
    std::filesystem::remove(file_path);
}

struct RangedMappingTestRegions {
    size_t offset_1;
    size_t size_1;
    size_t offset_2;
    size_t size_2;
    size_t file_size;
};
// regions, use file path (true) or file handle (false)
using RangedMappingTestParams = std::tuple<RangedMappingTestRegions, bool>;

namespace {
size_t calc_sector_actual_size(size_t offset, size_t size, size_t file_actual_size) {
    return size == auto_size ? file_actual_size - offset : size;
}
}  // namespace

class RangedMappingTest : public ::testing::TestWithParam<RangedMappingTestParams> {
protected:
    std::filesystem::path m_file_path;
    std::vector<char> m_sector_1, m_sector_2;

    void SetUp() override {
        const auto& [regions, use_file_path] = GetParam();
        const auto& [offset_1, size_1, offset_2, size_2, file_size] = regions;

        const auto actual_size_1 = calc_sector_actual_size(offset_1, size_1, file_size);
        const auto actual_size_2 = calc_sector_actual_size(offset_2, size_2, file_size);
        ASSERT_GT(file_size, 0);
        ASSERT_LE(offset_1 + actual_size_1, file_size);
        ASSERT_LE(offset_2 + actual_size_2, file_size);

        m_file_path = utils::generateTestFilePrefix();
        std::fstream s(m_file_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        ASSERT_TRUE(s.is_open());
        for (size_t i = 0; i < file_size; ++i) {
            s.put('_');
        }
        ASSERT_TRUE(s.good());
        s.seekp(offset_1);
        for (size_t i = 0; i < actual_size_1; ++i) {
            s.put('A' + i % 26);
        }
        s.seekp(offset_2);
        for (size_t i = 0; i < actual_size_2; ++i) {
            s.put('a' + i % 26);
        }
        // sectors may overlap, so read these after all
        m_sector_1.resize(actual_size_1);
        m_sector_2.resize(actual_size_2);
        s.seekg(offset_1);
        s.read(m_sector_1.data(), actual_size_1);
        s.seekg(offset_2);
        s.read(m_sector_2.data(), actual_size_2);
        ASSERT_TRUE(s.good());
    }

    void TearDown() override {
        std::filesystem::remove(m_file_path);
    }

public:
    static std::string test_name(const testing::TestParamInfo<RangedMappingTestParams>& info) {
        const auto& [regions, use_file_path] = info.param;
        const auto& [offset_1, size_1, offset_2, size_2, file_size] = regions;
        const auto actual_size_1 = calc_sector_actual_size(offset_1, size_1, file_size);
        const auto actual_size_2 = calc_sector_actual_size(offset_2, size_2, file_size);
        std::ostringstream ss;
        ss << "offset1_" << offset_1 << "_size1_" << std::to_string(actual_size_1) << "_offset2_" << offset_2
           << "_size2_" << std::to_string(actual_size_2) << "_file_size_" << std::to_string(file_size)
           << (use_file_path ? "_file_path" : "_file_handle");
        return ss.str();
    }
};

TEST_P(RangedMappingTest, compare_data) {
    const auto& [regions, use_file_path] = GetParam();
    const auto& [offset_1, size_1, offset_2, size_2, file_size] = regions;
    std::shared_ptr<MappedMemory> mm_1, mm_2;

    if (use_file_path) {
        mm_1 = load_mmap_object(m_file_path, offset_1, size_1);
        mm_2 = load_mmap_object(m_file_path, offset_2, size_2);
    } else {
        const auto handle_1 = utils::open_ro_file(m_file_path);
        const auto handle_2 = utils::open_ro_file(m_file_path);
        mm_1 = load_mmap_object(handle_1, offset_1, size_1);
        mm_2 = load_mmap_object(handle_2, offset_2, size_2);
    }
    ASSERT_NE(mm_1, nullptr);
    ASSERT_NE(mm_2, nullptr);

    EXPECT_EQ(mm_1->size(), m_sector_1.size());
    EXPECT_EQ(mm_2->size(), m_sector_2.size());
    EXPECT_EQ(m_sector_1, std::vector<char>(mm_1->data(), mm_1->data() + mm_1->size()));
    EXPECT_EQ(m_sector_2, std::vector<char>(mm_2->data(), mm_2->data() + mm_2->size()));
}

TEST_P(RangedMappingTest, compare_id) {
    const auto& [regions, use_file_path] = GetParam();
    const auto& [offset_1, size_1, offset_2, size_2, file_size] = regions;

    std::filesystem::path other_file_path = utils::generateTestFilePrefix();
    std::error_code ec;
    std::filesystem::copy_file(m_file_path, other_file_path, ec);
    ASSERT_FALSE(ec) << "Failed to copy file \"" << m_file_path << "\" for test setup: " << ec.message();

    {
        std::shared_ptr<MappedMemory> mm_1, mm_2, other_mm_1, other_mm_2, mm_1_;
        if (use_file_path) {
            mm_1 = load_mmap_object(m_file_path, offset_1, size_1);
            mm_2 = load_mmap_object(m_file_path, offset_2, size_2);
            other_mm_1 = load_mmap_object(other_file_path, offset_1, size_1);
            other_mm_2 = load_mmap_object(other_file_path, offset_2, size_2);
            mm_1_ = load_mmap_object(m_file_path, offset_1, size_1);
        } else {
            const auto handle = utils::open_ro_file(m_file_path);
            mm_1 = load_mmap_object(handle, offset_1, size_1);
            mm_2 = load_mmap_object(handle, offset_2, size_2);
            const auto other_handle = utils::open_ro_file(other_file_path);
            other_mm_1 = load_mmap_object(other_handle, offset_1, size_1);
            other_mm_2 = load_mmap_object(other_handle, offset_2, size_2);
            const auto handle_ = utils::open_ro_file(m_file_path);
            mm_1_ = load_mmap_object(handle_, offset_1, size_1);
        }

        ASSERT_NE(mm_1, nullptr);
        ASSERT_NE(mm_2, nullptr);
        ASSERT_NE(other_mm_1, nullptr);
        ASSERT_NE(other_mm_2, nullptr);
        ASSERT_NE(mm_1_, nullptr);

        EXPECT_NE(mm_1->get_id(), mm_2->get_id());
        EXPECT_NE(mm_1->get_id(), other_mm_1->get_id());
        EXPECT_NE(mm_2->get_id(), other_mm_2->get_id());
        EXPECT_EQ(mm_1->get_id(), mm_1_->get_id());
    }

    std::filesystem::remove(other_file_path);
}

static const auto pg_sz = []() {
    const auto sz = util::get_system_page_size();
    return sz > 0 ? static_cast<size_t>(sz) : size_t{4096};
}();
INSTANTIATE_TEST_SUITE_P(MappedMemory,
                         RangedMappingTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<RangedMappingTestRegions>{
                                                {pg_sz, 127, 2 * pg_sz, 101, (2 * pg_sz + 101)},
                                                {0, 3 * pg_sz, 7 * pg_sz, 1024, (7 * pg_sz + 1024)},
                                                {0, pg_sz, pg_sz, pg_sz, (2 * pg_sz)},
                                                {100, 50, 150, 30, 180},
                                                {1, auto_size, 0, auto_size, pg_sz},
                                                {0, 40, 0, 41, 42},
                                                {11, auto_size, 17, 80, 101},
                                                {10, auto_size, 0, 90, 100}}),
                                            ::testing::ValuesIn(std::vector<bool>{true, false})),
                         RangedMappingTest::test_name);

class HintEvictTest : public ::testing::Test {
protected:
    std::filesystem::path m_file_path;
    std::vector<uint8_t> m_expected;
    // 10 granules (10 × 64 KiB): partial_evict uses quarter = 160 KiB
    static constexpr size_t k_hint_evict_file_size = 64 * 1024 * 10;

    void SetUp() override {
        m_expected.resize(k_hint_evict_file_size);
        for (size_t i = 0; i < k_hint_evict_file_size; ++i) {
            // Prime-modulo pattern: easy to spot any byte corruption in failure output.
            m_expected[i] = static_cast<uint8_t>(i % 251);
        }
        m_file_path = utils::generateTestFilePrefix() + "_hint_evict";
        std::ofstream out(m_file_path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to create temp file: " << m_file_path;
        out.write(reinterpret_cast<const char*>(m_expected.data()), k_hint_evict_file_size);
        ASSERT_TRUE(out.good());
    }

    void TearDown() override {
        std::filesystem::remove(m_file_path);
    }

    static std::vector<uint8_t> read_mapped(MappedMemory& mm) {
        return {reinterpret_cast<uint8_t*>(mm.data()), reinterpret_cast<uint8_t*>(mm.data()) + mm.size()};
    }
};

TEST_F(HintEvictTest, full_evict_then_read_matches_original) {
    auto mm = load_mmap_object(m_file_path);
    ASSERT_NE(mm, nullptr);
    ASSERT_EQ(mm->size(), k_hint_evict_file_size);

    // Verify initial content before eviction.
    ASSERT_EQ(read_mapped(*mm), m_expected);

    // Evict all mapped pages.
    mm->hint_evict(0, auto_size);

    // All bytes must be transparently restored and unchanged.
    EXPECT_EQ(read_mapped(*mm), m_expected);
}

TEST_F(HintEvictTest, partial_evict_then_read_matches_original) {
    auto mm = load_mmap_object(m_file_path);
    ASSERT_NE(mm, nullptr);
    ASSERT_EQ(mm->size(), k_hint_evict_file_size);

    const size_t quarter = k_hint_evict_file_size / 4;
    mm->hint_evict(quarter, k_hint_evict_file_size / 2);

    EXPECT_EQ(read_mapped(*mm), m_expected);
}

TEST_F(HintEvictTest, multiple_evict_cycles_are_idempotent) {
    // Use a page-aligned but not granularity-aligned offset to exercise head_pad on each cycle.
    constexpr size_t k_offset = 4096;
    auto mm = load_mmap_object(m_file_path, k_offset, auto_size);
    ASSERT_NE(mm, nullptr);
    ASSERT_EQ(mm->size(), k_hint_evict_file_size - k_offset);

    const std::vector<uint8_t> expected_slice(m_expected.begin() + k_offset, m_expected.end());

    for (int cycle = 0; cycle < 3; ++cycle) {
        mm->hint_evict(0, auto_size);
        EXPECT_EQ(read_mapped(*mm), expected_slice) << "Data mismatch after evict cycle " << cycle;
    }
}

TEST_F(HintEvictTest, evict_then_read_via_file_handle_matches_original) {
    const auto handle = utils::open_ro_file(m_file_path);
    auto mm = load_mmap_object(handle, 0, auto_size);
    ASSERT_NE(mm, nullptr);
    ASSERT_EQ(mm->size(), k_hint_evict_file_size);

    mm->hint_evict(0, auto_size);

    EXPECT_EQ(read_mapped(*mm), m_expected);
}

TEST_F(HintEvictTest, evict_with_anonymous_tail_matches_original) {
    // Append extra bytes so the file size is not a multiple of the 64 KiB granularity.
    constexpr size_t k_extra = 4096;
    m_expected.resize(k_hint_evict_file_size + k_extra);
    for (size_t i = k_hint_evict_file_size; i < m_expected.size(); ++i)
        m_expected[i] = static_cast<uint8_t>(i % 251);
    {
        std::ofstream out(m_file_path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open());
        out.write(reinterpret_cast<const char*>(m_expected.data()), m_expected.size());
        ASSERT_TRUE(out.good());
    }

    auto mm = load_mmap_object(m_file_path);
    ASSERT_NE(mm, nullptr);
    ASSERT_EQ(mm->size(), m_expected.size());

    mm->hint_evict(0, auto_size);

    EXPECT_EQ(read_mapped(*mm), m_expected);
}

TEST_F(HintEvictTest, evict_with_nonzero_offset_matches_original) {
    // Use an offset that is page-aligned but NOT granularity-aligned.
    constexpr size_t k_offset = 4096;
    ASSERT_LT(k_offset, k_hint_evict_file_size);

    auto mm = load_mmap_object(m_file_path, k_offset, auto_size);
    ASSERT_NE(mm, nullptr);
    ASSERT_EQ(mm->size(), k_hint_evict_file_size - k_offset);

    const std::vector<uint8_t> expected_slice(m_expected.begin() + k_offset, m_expected.end());

    mm->hint_evict(0, auto_size);

    EXPECT_EQ(read_mapped(*mm), expected_slice);
}

class HintPrefetchTest : public ::testing::Test {
protected:
    std::filesystem::path m_file_path;

    void TearDown() override {
        std::filesystem::remove(m_file_path);
    }

    static std::vector<uint8_t> read_mapped(MappedMemory& mm) {
        return {reinterpret_cast<uint8_t*>(mm.data()), reinterpret_cast<uint8_t*>(mm.data()) + mm.size()};
    }
};

TEST_F(HintPrefetchTest, parallel_prefault_whole_file) {
    m_file_path = std::filesystem::path(utils::generateTestFilePrefix() + "_prefault_test.bin");
    constexpr size_t file_size = 5 * 1024 * 1024;  // 5 MiB (above 4 MiB threshold)
    std::vector<uint8_t> data(file_size);
    for (size_t i = 0; i < file_size; ++i)
        data[i] = static_cast<uint8_t>(i % 251);

    {
        std::ofstream f(m_file_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    {
        auto mapped = load_mmap_object(m_file_path);
        ASSERT_NE(mapped, nullptr);
        EXPECT_EQ(mapped->size(), file_size);

        EXPECT_NO_THROW(mapped->hint_prefetch());

        EXPECT_EQ(read_mapped(*mapped), data);
    }
}

TEST_F(HintPrefetchTest, parallel_prefault_partial_region) {
    m_file_path = std::filesystem::path(utils::generateTestFilePrefix() + "_prefault_partial.bin");
    constexpr size_t file_size = 8 * 1024 * 1024;  // 8 MB
    constexpr size_t prefault_offset = 1 * 1024 * 1024;
    constexpr size_t prefault_size = 5 * 1024 * 1024;
    std::vector<uint8_t> data(file_size);
    for (size_t i = 0; i < file_size; ++i)
        data[i] = static_cast<uint8_t>(i % 251);

    {
        std::ofstream f(m_file_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    {
        auto mapped = load_mmap_object(m_file_path);
        ASSERT_NE(mapped, nullptr);

        EXPECT_NO_THROW(mapped->hint_prefetch(prefault_offset, prefault_size));

        EXPECT_EQ(read_mapped(*mapped), data);
    }
}

TEST_F(HintPrefetchTest, parallel_prefault_below_threshold_is_noop) {
    m_file_path = std::filesystem::path(utils::generateTestFilePrefix() + "_prefault_small.bin");
    constexpr size_t file_size = 1024;  // 1 KB - below threshold
    std::vector<uint8_t> data(file_size);
    for (size_t i = 0; i < file_size; ++i)
        data[i] = static_cast<uint8_t>(i % 251);

    {
        std::ofstream f(m_file_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    {
        auto mapped = load_mmap_object(m_file_path);
        ASSERT_NE(mapped, nullptr);
        EXPECT_NO_THROW(mapped->hint_prefetch());  // no optimization
        EXPECT_EQ(read_mapped(*mapped), data);
    }
}

TEST_F(HintPrefetchTest, parallel_prefault_with_file_offset) {
    m_file_path = std::filesystem::path(utils::generateTestFilePrefix() + "_prefault_offset.bin");
    constexpr size_t file_size = 10 * 1024 * 1024;  // 10 MB
    constexpr size_t map_offset = 2 * 1024 * 1024;  // Map starting at 2 MB into file
    std::vector<uint8_t> data(file_size);
    for (size_t i = 0; i < file_size; ++i)
        data[i] = static_cast<uint8_t>(i % 251);

    {
        std::ofstream f(m_file_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    {
        auto mapped = load_mmap_object(m_file_path, map_offset);
        ASSERT_NE(mapped, nullptr);
        EXPECT_EQ(mapped->size(), file_size - map_offset);

        EXPECT_NO_THROW(mapped->hint_prefetch());

        EXPECT_EQ(read_mapped(*mapped), std::vector<uint8_t>(data.begin() + map_offset, data.end()));
    }
}

TEST_F(HintPrefetchTest, hint_prefetch_with_both_offsets) {
    m_file_path = std::filesystem::path(utils::generateTestFilePrefix() + "_prefault_both_offsets.bin");
    constexpr size_t file_size = 12 * 1024 * 1024;  // 12 MB
    constexpr size_t map_offset = 2 * 1024 * 1024;  // Map starting at 2 MB into file
    constexpr size_t pop_offset = 1 * 1024 * 1024;  // Populate starting at 1 MB into mapping
    constexpr size_t pop_size = 5 * 1024 * 1024;    // Populate 5 MB
    std::vector<uint8_t> data(file_size);
    for (size_t i = 0; i < file_size; ++i)
        data[i] = static_cast<uint8_t>(i % 251);

    {
        std::ofstream f(m_file_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    {
        auto mapped = load_mmap_object(m_file_path, map_offset);
        ASSERT_NE(mapped, nullptr);
        EXPECT_EQ(mapped->size(), file_size - map_offset);

        EXPECT_NO_THROW(mapped->hint_prefetch(pop_offset, pop_size));

        EXPECT_EQ(read_mapped(*mapped), std::vector<uint8_t>(data.begin() + map_offset, data.end()));
    }
}

// Investigates whether calling hint_prefetch(offset, size) and POSIX_FADV_SEQUENTIAL
// on a subregion of an already-cached file evicts pages *outside* that region
TEST_F(HintPrefetchTest, hint_prefetch_sequential_eviction_check) {
#ifndef __linux__
    GTEST_SKIP() << "utils::count_resident_pages is not implemented on this platform yet CVS-186579";
#endif
    constexpr size_t file_size = 128 * 1024 * 1024;

    constexpr size_t prefetch_offset = 80 * 1024 * 1024;
    constexpr size_t prefetch_size = 16 * 1024 * 1024;

    constexpr size_t prefix_mb = 64;
    constexpr size_t prefix_size = prefix_mb * 1024 * 1024;

    const size_t page = static_cast<size_t>(util::get_system_page_size());
    const size_t total_prefix_pages = prefix_size / page;

    m_file_path = std::filesystem::path(utils::generateTestFilePrefix() + "_file.bin");
    {
        std::vector<uint8_t> data(file_size);
        for (size_t i = 0; i < file_size; ++i)
            data[i] = static_cast<uint8_t>(i % 251);
        std::ofstream f(m_file_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    auto mapped = load_mmap_object(m_file_path);
    volatile char sink = 0;
    for (size_t i = 0; i < prefix_size; i += page) {
        sink += mapped->data()[i];
    }
    const size_t pages_before = utils::count_resident_pages(mapped->data(), prefix_size);
    ASSERT_EQ(pages_before, total_prefix_pages)
        << "Expected all " << total_prefix_pages << " prefix pages resident after warmup, but found " << pages_before;

    mapped->hint_prefetch(prefetch_offset, prefetch_size);
    const size_t pages_after = utils::count_resident_pages(mapped->data(), prefix_size);
    EXPECT_EQ(pages_after, pages_before) << "hint_prefetch evicted pages.";
}

}  // namespace ov::test
