// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::test {

// Verify the constexpr contract at compile time for the most common alignments.
static_assert(ov::util::align_size_up(0, 64) == 0);
static_assert(ov::util::align_size_up(1, 64) == 64);
static_assert(ov::util::align_size_up(64, 64) == 64);
static_assert(ov::util::align_size_up(65, 64) == 128);
static_assert(ov::util::align_size_up(100, 16) == 112);
static_assert(ov::util::align_size_up(100, 1) == 100);
static_assert(ov::util::align_size_up(7, 8) == 8);
static_assert(ov::util::align_size_up(8, 8) == 8);
static_assert(ov::util::align_size_up(9, 8) == 16);
static_assert(ov::util::align_size_up(9, alignof(std::max_align_t)) == 16);

static_assert(ov::util::align_size_down(0, 64) == 0);
static_assert(ov::util::align_size_down(63, 64) == 0);
static_assert(ov::util::align_size_down(64, 64) == 64);
static_assert(ov::util::align_size_down(65, 64) == 64);
static_assert(ov::util::align_size_down(100, 16) == 96);
static_assert(ov::util::align_size_down(100, 1) == 100);

static_assert(ov::util::align_region(64, 32, 64).m_address == 64);
static_assert(ov::util::align_region(64, 32, 64).m_length == 32);
static_assert(ov::util::align_region(64, 32, 64).m_gap == 0);
static_assert(ov::util::align_region(65, 32, 64).m_address == 64);
static_assert(ov::util::align_region(65, 32, 64).m_length == 33);
static_assert(ov::util::align_region(65, 32, 64).m_gap == 1);

using AlignSizeUpTest = testing::Test;

TEST_F(AlignSizeUpTest, already_aligned_value_is_unchanged) {
    EXPECT_EQ(64u, util::align_size_up(64, 64));
    EXPECT_EQ(128u, util::align_size_up(128, 64));
    EXPECT_EQ(16u, util::align_size_up(16, 16));
}

TEST_F(AlignSizeUpTest, unaligned_value_is_rounded_up) {
    EXPECT_EQ(64u, util::align_size_up(1, 64));
    EXPECT_EQ(64u, util::align_size_up(63, 64));
    EXPECT_EQ(128u, util::align_size_up(65, 64));
    EXPECT_EQ(112u, util::align_size_up(100, 16));
}

TEST_F(AlignSizeUpTest, alignment_1_never_rounds) {
    EXPECT_EQ(0u, util::align_size_up(0, 1));
    EXPECT_EQ(100u, util::align_size_up(100, 1));
    EXPECT_EQ(255u, util::align_size_up(255, 1));
}

TEST_F(AlignSizeUpTest, zero_size_returns_zero) {
    EXPECT_EQ(0u, util::align_size_up(0, 64));
    EXPECT_EQ(0u, util::align_size_up(0, 4096));
}

using AlignSizeDownTest = testing::Test;

TEST_F(AlignSizeDownTest, already_aligned_value_is_unchanged) {
    EXPECT_EQ(64u, util::align_size_down(64, 64));
    EXPECT_EQ(128u, util::align_size_down(128, 64));
    EXPECT_EQ(16u, util::align_size_down(16, 16));
}

TEST_F(AlignSizeDownTest, unaligned_value_is_rounded_down) {
    EXPECT_EQ(0u, util::align_size_down(1, 64));
    EXPECT_EQ(0u, util::align_size_down(63, 64));
    EXPECT_EQ(64u, util::align_size_down(65, 64));
    EXPECT_EQ(96u, util::align_size_down(100, 16));
}

TEST_F(AlignSizeDownTest, alignment_1_never_rounds) {
    EXPECT_EQ(0u, util::align_size_down(0, 1));
    EXPECT_EQ(100u, util::align_size_down(100, 1));
    EXPECT_EQ(255u, util::align_size_down(255, 1));
}

TEST_F(AlignSizeDownTest, zero_size_returns_zero) {
    EXPECT_EQ(0u, util::align_size_down(0, 64));
    EXPECT_EQ(0u, util::align_size_down(0, 4096));
}

using AlignRegionTest = testing::Test;

TEST_F(AlignRegionTest, aligned_base_has_zero_gap) {
    const auto r = util::align_region(64, 32, 64);
    EXPECT_EQ(64u, r.m_address);
    EXPECT_EQ(32u, r.m_length);
    EXPECT_EQ(0u, r.m_gap);
}

TEST_F(AlignRegionTest, unaligned_base_is_rounded_down) {
    const auto r = util::align_region(65, 32, 64);
    EXPECT_EQ(64u, r.m_address);
    EXPECT_EQ(33u, r.m_length);
    EXPECT_EQ(1u, r.m_gap);
}

TEST_F(AlignRegionTest, gap_equals_base_minus_address) {
    const auto r = util::align_region(100, 50, 64);
    EXPECT_EQ(64u, r.m_address);
    EXPECT_EQ(86u, r.m_length);
    EXPECT_EQ(36u, r.m_gap);
}

TEST_F(AlignRegionTest, result_covers_original_range) {
    for (uintptr_t base : {0u, 1u, 63u, 64u, 65u, 100u, 127u}) {
        const auto r = util::align_region(base, 128, 64);
        EXPECT_LE(r.m_address, base) << "base=" << base;
        EXPECT_GE(r.m_address + r.m_length, base + 128u) << "base=" << base;
    }
}

using AlignedAllocTest = testing::Test;

TEST_F(AlignedAllocTest, returns_non_null_for_valid_args) {
    void* ptr = util::aligned_alloc(128, 64);
    ASSERT_NE(nullptr, ptr);
    util::aligned_free(ptr);
}

TEST_F(AlignedAllocTest, pointer_satisfies_requested_alignment) {
    for (auto align : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u}) {
        void* ptr = util::aligned_alloc(256, align);
        ASSERT_NE(nullptr, ptr) << "align=" << align;
        EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(ptr) % align) << "align=" << align;
        util::aligned_free(ptr);
    }
}

TEST_F(AlignedAllocTest, zero_alignment_uses_default_alignment) {
    void* ptr = util::aligned_alloc(64, 0);
    ASSERT_NE(nullptr, ptr);
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(ptr) % alignof(std::max_align_t));
    util::aligned_free(ptr);
}

TEST_F(AlignedAllocTest, free_nullptr_is_noop) {
    EXPECT_NO_FATAL_FAILURE(util::aligned_free(nullptr));
}

class VmPrefetchMappedFileTest : public testing::TestWithParam<size_t> {
protected:
    std::filesystem::path m_file_path;

    void TearDown() override {
        std::filesystem::remove(m_file_path);
    }
};

TEST_P(VmPrefetchMappedFileTest, prefetch_faults_in_mapped_file_and_preserves_data) {
    const size_t num_threads = GetParam();
    const size_t size = 64 * util::min_page_alignment;

    std::vector<uint8_t> expected(size);
    for (size_t i = 0; i < size; ++i)
        expected[i] = static_cast<uint8_t>(i % 251);

    m_file_path = utils::generateTestFilePrefix() + "_vm_prefetch.bin";
    {
        std::ofstream f(m_file_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(expected.data()), expected.size());
    }

    auto mapped = load_mmap_object(m_file_path);
    ASSERT_NE(mapped, nullptr);
    ASSERT_EQ(mapped->size(), size);

    EXPECT_NO_FATAL_FAILURE(util::vm_prefetch(mapped->data(), mapped->size(), num_threads));

    const std::vector<uint8_t> actual(reinterpret_cast<uint8_t*>(mapped->data()),
                                      reinterpret_cast<uint8_t*>(mapped->data()) + mapped->size());
    EXPECT_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(NumThreads, VmPrefetchMappedFileTest, testing::Values(0u, 5u, 10u));

}  // namespace ov::test
