// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/mmap_object.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
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
        mm_1 = load_mmap_object_from_handle(handle_1, offset_1, size_1);
        mm_2 = load_mmap_object_from_handle(handle_2, offset_2, size_2);
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

    std::shared_ptr<MappedMemory> mm_1, mm_2, other_mm_1, other_mm_2, mm_1_;
    if (use_file_path) {
        mm_1 = load_mmap_object(m_file_path, offset_1, size_1);
        mm_2 = load_mmap_object(m_file_path, offset_2, size_2);
        other_mm_1 = load_mmap_object(other_file_path, offset_1, size_1);
        other_mm_2 = load_mmap_object(other_file_path, offset_2, size_2);
        mm_1_ = load_mmap_object(m_file_path, offset_1, size_1);
    } else {
        const auto handle_1 = utils::open_ro_file(m_file_path);
        const auto handle_2 = utils::open_ro_file(m_file_path);
        const auto other_handle = utils::open_ro_file(other_file_path);
        mm_1 = load_mmap_object_from_handle(handle_1, offset_1, size_1);
        mm_2 = load_mmap_object_from_handle(handle_2, offset_2, size_2);
        other_mm_1 = load_mmap_object_from_handle(other_handle, offset_1, size_1);
        other_mm_2 = load_mmap_object_from_handle(other_handle, offset_2, size_2);
        mm_1_ = load_mmap_object_from_handle(handle_1, offset_1, size_1);
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

}  // namespace ov::test
