// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#ifdef OV_GPU_WITH_SYCL_RT

#include "sycl_test_context.hpp"

#include "intel_gpu/runtime/memory.hpp"
#include "runtime/sycl/sycl_memory.hpp"

#include <numeric>
#include <vector>

using namespace cldnn;
using namespace sycl_tests;

namespace {

std::vector<allocation_type> get_supported_sycl_alloc_types(const std::shared_ptr<engine>& sycl_test_engine) {
    std::vector<allocation_type> alloc_types;
    if (sycl_test_engine->supports_allocation(allocation_type::usm_host)) {
        alloc_types.push_back(allocation_type::usm_host);
    }
    if (sycl_test_engine->supports_allocation(allocation_type::usm_shared)) {
        alloc_types.push_back(allocation_type::usm_shared);
    }
    if (sycl_test_engine->supports_allocation(allocation_type::usm_device)) {
        alloc_types.push_back(allocation_type::usm_device);
    }

    return alloc_types;
}

} // namespace

TEST(sycl_usm_memory, copy_and_read_buffer) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.empty()) {
        GTEST_SKIP() << "No supported SYCL USM allocation types";
    }

    const size_t values_count = 100;
    const layout linear_layout = {{1, 1, static_cast<int32_t>(values_count), 1}, data_types::f32, format::bfyx};
    std::vector<float> src_buffer(values_count);
    std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

    for (auto alloc_type : alloc_types) {
        SCOPED_TRACE(static_cast<int>(alloc_type));
        auto mem = ctx.sycl_test_engine->allocate_memory(linear_layout, alloc_type);
        ASSERT_NE(mem, nullptr);

        OV_ASSERT_NO_THROW(mem->copy_from(*ctx.sycl_test_stream, src_buffer.data(), true));

        std::vector<float> dst_buffer(values_count, -1.0f);
        OV_ASSERT_NO_THROW(mem->copy_to(*ctx.sycl_test_stream, dst_buffer.data(), true));
        ASSERT_EQ(src_buffer, dst_buffer);
    }
}

TEST(sycl_usm_memory, offset_copy_between_memories) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.empty()) {
        GTEST_SKIP() << "No supported SYCL USM allocation types";
    }

    constexpr size_t src_offset = 37;
    constexpr size_t dst_offset = 53;
    constexpr size_t copy_size = 381;
    const size_t src_size = src_offset + copy_size;
    const size_t dst_size = dst_offset + copy_size;

    const layout src_layout = {{static_cast<int64_t>(src_size)}, data_types::u8, format::bfyx};
    const layout dst_layout = {{static_cast<int64_t>(dst_size)}, data_types::u8, format::bfyx};

    std::vector<uint8_t> src_data(src_size);
    for (size_t i = 0; i < src_size; i++) {
        src_data[i] = static_cast<uint8_t>(i % 64);
    }

    for (auto src_alloc : alloc_types) {
        for (auto dst_alloc : alloc_types) {
            SCOPED_TRACE(static_cast<int>(src_alloc));
            SCOPED_TRACE(static_cast<int>(dst_alloc));

            auto src_mem = ctx.sycl_test_engine->allocate_memory(src_layout, src_alloc);
            auto dst_mem = ctx.sycl_test_engine->allocate_memory(dst_layout, dst_alloc);
            ASSERT_NE(src_mem, nullptr);
            ASSERT_NE(dst_mem, nullptr);

            OV_ASSERT_NO_THROW(src_mem->copy_from(*ctx.sycl_test_stream, src_data.data(), 0, 0, src_size, true));
            OV_ASSERT_NO_THROW(dst_mem->copy_from(*ctx.sycl_test_stream, *src_mem, src_offset, dst_offset, copy_size, true));

            std::vector<uint8_t> actual(copy_size, 0);
            OV_ASSERT_NO_THROW(dst_mem->copy_to(*ctx.sycl_test_stream, actual.data(), dst_offset, 0, copy_size, true));

            for (size_t i = 0; i < copy_size; i++) {
                ASSERT_EQ(src_data[i + src_offset], actual[i]);
            }
        }
    }
}

TEST(sycl_usm_memory, offset_copy_host_roundtrip) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.empty()) {
        GTEST_SKIP() << "No supported SYCL USM allocation types";
    }

    constexpr size_t src_offset = 79;
    constexpr size_t dst_offset = 113;
    constexpr size_t copy_size = 257;
    const size_t host_size = src_offset + copy_size;
    const size_t mem_size = dst_offset + copy_size;

    const layout mem_layout = {{static_cast<int64_t>(mem_size)}, data_types::u8, format::bfyx};
    std::vector<uint8_t> src_buffer(host_size);
    std::vector<uint8_t> dst_buffer(host_size, 0);
    for (size_t i = 0; i < host_size; i++) {
        src_buffer[i] = static_cast<uint8_t>(i % 64);
    }

    for (auto alloc_type : alloc_types) {
        SCOPED_TRACE(static_cast<int>(alloc_type));
        auto mem = ctx.sycl_test_engine->allocate_memory(mem_layout, alloc_type);
        ASSERT_NE(mem, nullptr);

        OV_ASSERT_NO_THROW(mem->copy_from(*ctx.sycl_test_stream, src_buffer.data(), src_offset, dst_offset, copy_size, true));
        OV_ASSERT_NO_THROW(mem->copy_to(*ctx.sycl_test_stream, dst_buffer.data(), dst_offset, src_offset, copy_size, true));

        for (size_t i = 0; i < copy_size; i++) {
            ASSERT_EQ(src_buffer[src_offset + i], dst_buffer[src_offset + i]);
        }
    }
}

TEST(sycl_usm_memory, allocate_different_types) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.empty()) {
        GTEST_SKIP() << "No supported SYCL USM allocation types";
    }

    const layout test_layout = {{100}, data_types::f32, format::bfyx};
    for (auto alloc_type : alloc_types) {
        SCOPED_TRACE(static_cast<int>(alloc_type));
        auto mem = ctx.sycl_test_engine->allocate_memory(test_layout, alloc_type);
        ASSERT_NE(mem, nullptr);
        ASSERT_EQ(mem->get_allocation_type(), alloc_type);
        ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::gpu_usm>(mem), nullptr);
    }
}

TEST(sycl_usm_memory, fill_buffer) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.empty()) {
        GTEST_SKIP() << "No supported SYCL USM allocation types";
    }

    const size_t values_count = 100;
    const layout linear_layout = {{1, 1, static_cast<int32_t>(values_count), 1}, data_types::u8, format::bfyx};
    const unsigned char fill_pattern = 42;

    for (auto alloc_type : alloc_types) {
        SCOPED_TRACE(static_cast<int>(alloc_type));
        auto mem = ctx.sycl_test_engine->allocate_memory(linear_layout, alloc_type);
        ASSERT_NE(mem, nullptr);

        OV_ASSERT_NO_THROW(mem->fill(*ctx.sycl_test_stream, fill_pattern, {}, true));

        if (alloc_type == allocation_type::usm_host || alloc_type == allocation_type::usm_shared) {
            auto usm_mem = std::dynamic_pointer_cast<cldnn::sycl::gpu_usm>(mem);
            ASSERT_NE(usm_mem, nullptr);
            auto ptr = usm_mem->buffer_ptr();
            for (size_t i = 0; i < values_count; i++) {
                ASSERT_EQ(*(static_cast<unsigned char*>(ptr) + i), fill_pattern);
            }
        } else if (alloc_type == allocation_type::usm_device) {
            std::vector<uint8_t> host_buffer(values_count);
            OV_ASSERT_NO_THROW(mem->copy_to(*ctx.sycl_test_stream, host_buffer.data(), true));
            for (size_t i = 0; i < values_count; i++) {
                ASSERT_EQ(host_buffer[i], fill_pattern);
            }
        }
    }
}

TEST(sycl_usm_memory, copy_between_different_alloc_types) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.size() < 2) {
        GTEST_SKIP() << "Need at least 2 supported SYCL USM allocation types";
    }

    const size_t values_count = 100;
    const layout linear_layout = {{1, 1, static_cast<int32_t>(values_count), 1}, data_types::f32, format::bfyx};
    std::vector<float> src_buffer(values_count);
    std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

    auto src_mem = ctx.sycl_test_engine->allocate_memory(linear_layout, alloc_types[0]);
    ASSERT_NE(src_mem, nullptr);
    OV_ASSERT_NO_THROW(src_mem->copy_from(*ctx.sycl_test_stream, src_buffer.data(), true));

    for (size_t i = 1; i < alloc_types.size(); i++) {
        SCOPED_TRACE(static_cast<int>(alloc_types[i]));
        auto dst_mem = ctx.sycl_test_engine->allocate_memory(linear_layout, alloc_types[i]);
        ASSERT_NE(dst_mem, nullptr);

        OV_ASSERT_NO_THROW(dst_mem->copy_from(*ctx.sycl_test_stream, *src_mem, true));

        std::vector<float> result(values_count);
        OV_ASSERT_NO_THROW(dst_mem->copy_to(*ctx.sycl_test_stream, result.data(), true));
        ASSERT_EQ(src_buffer, result);
    }
}

TEST(sycl_usm_memory, copy_from_sycl_buffer_to_usm_alloc_types) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.empty()) {
        GTEST_SKIP() << "No supported SYCL USM allocation types";
    }
    if (!ctx.sycl_test_engine->supports_allocation(allocation_type::sycl_buffer)) {
        GTEST_SKIP() << "sycl_buffer allocation is not supported on this device";
    }

    constexpr size_t src_offset = 19;
    constexpr size_t dst_offset = 31;
    constexpr size_t copy_size = 257;
    const size_t src_size = src_offset + copy_size;
    const size_t dst_size = dst_offset + copy_size;

    const layout src_layout = {{static_cast<int64_t>(src_size)}, data_types::u8, format::bfyx};
    const layout dst_layout = {{static_cast<int64_t>(dst_size)}, data_types::u8, format::bfyx};

    auto src_mem = ctx.sycl_test_engine->allocate_memory(src_layout, allocation_type::sycl_buffer);
    ASSERT_NE(src_mem, nullptr);

    std::vector<uint8_t> src_data(src_size);
    for (size_t i = 0; i < src_size; i++) {
        src_data[i] = static_cast<uint8_t>((i * 37 + 11) % 251);
    }
    OV_ASSERT_NO_THROW(src_mem->copy_from(*ctx.sycl_test_stream, src_data.data(), 0, 0, src_size, true));

    for (auto dst_alloc : alloc_types) {
        SCOPED_TRACE(static_cast<int>(dst_alloc));

        auto dst_mem = ctx.sycl_test_engine->allocate_memory(dst_layout, dst_alloc);
        ASSERT_NE(dst_mem, nullptr);

        OV_ASSERT_NO_THROW(dst_mem->copy_from(*ctx.sycl_test_stream, *src_mem, src_offset, dst_offset, copy_size, true));

        std::vector<uint8_t> actual(copy_size, 0);
        OV_ASSERT_NO_THROW(dst_mem->copy_to(*ctx.sycl_test_stream, actual.data(), dst_offset, 0, copy_size, true));

        for (size_t i = 0; i < copy_size; i++) {
            ASSERT_EQ(src_data[src_offset + i], actual[i]);
        }
    }
}

TEST(sycl_usm_memory, copy_small_to_large_buffer) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.empty()) {
        GTEST_SKIP() << "No supported SYCL USM allocation types";
    }

    const size_t small_size = 512;
    const size_t large_size = 768;

    const layout small_layout = {{static_cast<int64_t>(small_size)}, data_types::u8, format::bfyx};
    const layout large_layout = {{static_cast<int64_t>(large_size)}, data_types::u8, format::bfyx};

    auto small_mem = ctx.sycl_test_engine->allocate_memory(small_layout, alloc_types[0]);
    auto large_mem = ctx.sycl_test_engine->allocate_memory(large_layout, alloc_types[0]);
    ASSERT_NE(small_mem, nullptr);
    ASSERT_NE(large_mem, nullptr);

    const uint8_t fill_pattern = 0xAB;
    OV_ASSERT_NO_THROW(small_mem->fill(*ctx.sycl_test_stream, fill_pattern, {}, true));
    OV_ASSERT_NO_THROW(large_mem->fill(*ctx.sycl_test_stream, 0, {}, true));

    OV_ASSERT_NO_THROW(small_mem->copy_to(*ctx.sycl_test_stream, *large_mem, true));

    std::vector<uint8_t> host_buffer(large_size, 0xFF);
    OV_ASSERT_NO_THROW(large_mem->copy_to(*ctx.sycl_test_stream, host_buffer.data(), true));
    for (size_t i = 0; i < small_size; i++) {
        ASSERT_EQ(host_buffer[i], fill_pattern);
    }
    for (size_t i = small_size; i < large_size; i++) {
        ASSERT_EQ(host_buffer[i], 0);
    }
}

TEST(sycl_usm_memory, lock_zero_size_usm_device) {
    auto ctx = create_sycl_test_context();
    if (!ctx.sycl_test_engine->supports_allocation(allocation_type::usm_device)) {
        GTEST_SKIP() << "SYCL usm_device allocation is not supported";
    }

    const layout zero_layout = {{0}, data_types::u8, format::bfyx};
    auto mem = ctx.sycl_test_engine->allocate_memory(zero_layout, allocation_type::usm_device);
    ASSERT_NE(mem, nullptr);

    auto usm_mem = std::dynamic_pointer_cast<cldnn::sycl::gpu_usm>(mem);
    ASSERT_NE(usm_mem, nullptr);

    void *ptr0, *ptr1;
    OV_ASSERT_NO_THROW(ptr0 = usm_mem->lock(*ctx.sycl_test_stream, mem_lock_type::write));
    OV_ASSERT_NO_THROW(ptr1 = usm_mem->lock(*ctx.sycl_test_stream, mem_lock_type::read));
    ASSERT_EQ(ptr0, nullptr);
    ASSERT_EQ(ptr1, nullptr);

    OV_ASSERT_NO_THROW(usm_mem->unlock(*ctx.sycl_test_stream));
    OV_ASSERT_NO_THROW(usm_mem->unlock(*ctx.sycl_test_stream));
}

TEST(sycl_usm_memory, subbuffer_keeps_allocation_alive_after_parent_release) {
    auto ctx = create_sycl_test_context();
    auto alloc_types = get_supported_sycl_alloc_types(ctx.sycl_test_engine);
    if (alloc_types.empty()) {
        GTEST_SKIP() << "No supported SYCL USM allocation types";
    }

    constexpr size_t parent_size = 512;
    constexpr size_t sub_offset = 96;
    constexpr size_t sub_size = 256;

    const layout parent_layout = {{static_cast<int64_t>(parent_size)}, data_types::u8, format::bfyx};
    const layout sub_layout = {{static_cast<int64_t>(sub_size)}, data_types::u8, format::bfyx};

    std::vector<uint8_t> src(parent_size);
    for (size_t i = 0; i < parent_size; i++) {
        src[i] = static_cast<uint8_t>((i * 13 + 7) % 251);
    }

    for (auto alloc_type : alloc_types) {
        SCOPED_TRACE(static_cast<int>(alloc_type));

        auto parent_mem = ctx.sycl_test_engine->allocate_memory(parent_layout, alloc_type);
        ASSERT_NE(parent_mem, nullptr);
        OV_ASSERT_NO_THROW(parent_mem->copy_from(*ctx.sycl_test_stream, src.data(), true));

        auto sub_mem = ctx.sycl_test_engine->create_subbuffer(*parent_mem, sub_layout, sub_offset);
        ASSERT_NE(sub_mem, nullptr);

        parent_mem.reset();

        // Stress allocator reuse paths after parent release to make lifetime issues visible.
        for (size_t i = 0; i < 4; i++) {
            auto tmp = ctx.sycl_test_engine->allocate_memory(parent_layout, alloc_type);
            ASSERT_NE(tmp, nullptr);
            OV_ASSERT_NO_THROW(tmp->fill(*ctx.sycl_test_stream, static_cast<unsigned char>(i + 1), {}, true));
        }

        std::vector<uint8_t> actual(sub_size, 0);
        OV_ASSERT_NO_THROW(sub_mem->copy_to(*ctx.sycl_test_stream, actual.data(), true));

        for (size_t i = 0; i < sub_size; i++) {
            ASSERT_EQ(src[sub_offset + i], actual[i]);
        }
    }
}

#endif  // OV_GPU_WITH_SYCL_RT
