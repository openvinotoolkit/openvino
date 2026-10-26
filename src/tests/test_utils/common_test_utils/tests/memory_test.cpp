// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/memory.hpp"

#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

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

static_assert(ov::util::align_size_up_overflow(0, 64) == 0);
static_assert(ov::util::align_size_up_overflow(65, 64) == 128);
static_assert(ov::util::align_size_up_overflow(std::numeric_limits<size_t>::max(), 1) ==
              std::numeric_limits<size_t>::max());
static_assert(!ov::util::align_size_up_overflow(std::numeric_limits<size_t>::max() - 1, 8).has_value());
static_assert(!ov::util::align_size_up_overflow(std::numeric_limits<size_t>::max() - 62, 64).has_value());
static_assert(ov::util::align_size_up_overflow(std::numeric_limits<size_t>::max() - 63, 64) ==
              std::numeric_limits<size_t>::max() - 63);

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

static_assert(ov::util::MemoryView{}.empty());
static_assert(ov::util::MemoryView{}.data() == nullptr);
static_assert(ov::util::MemoryView{}.size() == 0);

constexpr std::byte memory_view_bytes[4]{};
static_assert(ov::util::MemoryView{nullptr, 0}.empty());
static_assert(ov::util::MemoryView{nullptr, sizeof(memory_view_bytes)}.empty());
static_assert(ov::util::MemoryView{memory_view_bytes, 0}.empty());
static_assert(!ov::util::MemoryView{memory_view_bytes, sizeof(memory_view_bytes)}.empty());

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

TEST_F(AlignedAllocTest, small_alignment_uses_max_align_t) {
    for (size_t align = 1; align < alignof(std::max_align_t); align <<= 1) {
        void* ptr = util::aligned_alloc(3, align);
        ASSERT_NE(nullptr, ptr) << "align=" << align;
        EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(ptr) % alignof(std::max_align_t)) << "align=" << align;
        util::aligned_free(ptr);
    }
}

TEST_F(AlignedAllocTest, size_too_big_for_alignment_returns_null) {
    constexpr auto max_size = std::numeric_limits<size_t>::max();
    for (auto align : {size_t{0}, size_t{1}, alignof(std::max_align_t), size_t{64}}) {
        EXPECT_EQ(nullptr, util::aligned_alloc(max_size - 1, align)) << "align=" << align;
    }
}

TEST_F(AlignedAllocTest, free_nullptr_is_noop) {
    EXPECT_NO_FATAL_FAILURE(util::aligned_free(nullptr));
}

using MemoryViewTest = testing::Test;

TEST_F(MemoryViewTest, default_constructed_view_is_empty) {
    const util::MemoryView view;
    EXPECT_TRUE(view.empty());
    EXPECT_EQ(nullptr, view.data());
    EXPECT_EQ(0u, view.size());
}

TEST_F(MemoryViewTest, zero_size_with_valid_pointer_is_empty) {
    const std::byte b{};
    const util::MemoryView view{&b, 0};
    EXPECT_TRUE(view.empty());
}

TEST_F(MemoryViewTest, non_zero_size_with_nullptr_is_empty) {
    const util::MemoryView view{nullptr, 4};
    EXPECT_TRUE(view.empty());
}

TEST_F(MemoryViewTest, non_zero_size_with_valid_pointer_is_not_empty) {
    const std::array<std::byte, 4> bytes{};
    const util::MemoryView view{bytes.data(), 4};
    EXPECT_FALSE(view.empty());
    EXPECT_EQ(bytes.data(), view.data());
    EXPECT_EQ(4u, view.size());
    EXPECT_EQ(view.data(), view.begin());
    EXPECT_EQ(view.data() + view.size(), view.end());
}

TEST_F(MemoryViewTest, as_memory_view_spans_object) {
    const int32_t value = 42;
    const auto view = util::as_memory_view(value);
    EXPECT_EQ(reinterpret_cast<const std::byte*>(&value), view.data());
    EXPECT_EQ(sizeof(value), view.size());
    EXPECT_FALSE(view.empty());
}

TEST_F(MemoryViewTest, as_memory_view_exposes_object_representation) {
    const uint32_t value = 0x01020304;
    const auto view = util::as_memory_view(value);
    EXPECT_EQ(0, std::memcmp(view.data(), &value, sizeof(value)));
}

TEST_F(MemoryViewTest, as_memory_view_is_non_owning) {
    uint32_t value = 1;
    const auto view = util::as_memory_view(value);
    value = 0xAABBCCDD;
    EXPECT_EQ(0, std::memcmp(view.data(), &value, sizeof(value)));
}

TEST_F(MemoryViewTest, as_memory_view_of_struct_covers_padding) {
    struct Padded {
        char c;
        int64_t i;
    };
    const Padded value{};
    EXPECT_EQ(sizeof(Padded), util::as_memory_view(value).size());
}

TEST_F(MemoryViewTest, as_memory_view_of_array_covers_all_elements) {
    const std::array<uint16_t, 4> arr{1, 2, 3, 4};
    const auto view = util::as_memory_view(arr);
    EXPECT_EQ(sizeof(arr), view.size());
    EXPECT_EQ(reinterpret_cast<const std::byte*>(arr.data()), view.data());
}

template <typename T, typename = void>
struct CanMakeMemoryView : std::false_type {};

template <typename T>
struct CanMakeMemoryView<T, std::void_t<decltype(util::as_memory_view(std::declval<T>()))>> : std::true_type {};

TEST_F(MemoryViewTest, as_memory_view_rejects_rvalues) {
    static_assert(CanMakeMemoryView<int&>::value);
    static_assert(CanMakeMemoryView<const int&>::value);
    static_assert(!CanMakeMemoryView<int>::value);
    static_assert(!CanMakeMemoryView<const int>::value);
}

}  // namespace ov::test
