// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_test_context.hpp"

#include "intel_gpu/runtime/memory.hpp"
#include "ze/ze_memory.hpp"

#include <numeric>
#include <vector>

using namespace cldnn;
using namespace ze_tests;

namespace {

struct ImageLayoutTestParams {
    layout image_layout;
    std::string name;
};

// Tests below expect u8 data type
const std::vector<ImageLayoutTestParams> all_image_layouts = {
    { {ov::PartialShape{1, 4, 8, 1}, data_types::u8, format::image_2d_rgba}, "rgba_8x4" },
    { {ov::PartialShape{1, 4, 8, 1}, data_types::u8, format::nv12}, "nv12_y_8x4" },
    { {ov::PartialShape{1, 4, 8, 2}, data_types::u8, format::nv12}, "nv12_uv_8x4" },
};

}  // namespace

class ze_image_memory_tests : public testing::TestWithParam<ImageLayoutTestParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<ImageLayoutTestParams>& info) {
        return info.param.name;
    }

protected:
    void SetUp() override {
        ctx = create_ze_test_context();
        if (!ctx.ze_test_engine->supports_allocation(allocation_type::ze_image)) {
            GTEST_SKIP() << "ZE image allocation not supported";
        }
    }

    ze_test_context ctx;
};

TEST_P(ze_image_memory_tests, can_allocate) {
    const auto& image_layout = GetParam().image_layout;
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    ASSERT_NE(mem, nullptr);
    EXPECT_EQ(mem->get_allocation_type(), allocation_type::ze_image);
    EXPECT_EQ(mem->size(), image_layout.bytes_count());
    EXPECT_NE(std::dynamic_pointer_cast<ze::gpu_image2d>(mem), nullptr);
}

TEST_P(ze_image_memory_tests, can_copy_from_host_and_copy_to_host) {
    const auto& image_layout = GetParam().image_layout;
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    ASSERT_NE(mem, nullptr);

    const size_t bytes = mem->size();
    std::vector<uint8_t> src(bytes);
    std::iota(src.begin(), src.end(), uint8_t{0});

    OV_ASSERT_NO_THROW(mem->copy_from(*ctx.ze_test_stream, src.data(), true));

    std::vector<uint8_t> dst(bytes, 0xFF);
    OV_ASSERT_NO_THROW(mem->copy_to(*ctx.ze_test_stream, dst.data(), true));

    EXPECT_EQ(src, dst);
}

TEST_P(ze_image_memory_tests, can_fill) {
    const auto& image_layout = GetParam().image_layout;
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    ASSERT_NE(mem, nullptr);

    const unsigned char pattern = 0xAB;
    OV_ASSERT_NO_THROW(mem->fill(*ctx.ze_test_stream, pattern, {}, true));

    const size_t bytes = mem->size();
    std::vector<uint8_t> readback(bytes, 0x00);
    OV_ASSERT_NO_THROW(mem->copy_to(*ctx.ze_test_stream, readback.data(), true));

    for (size_t i = 0; i < bytes; i++) {
        EXPECT_EQ(readback[i], pattern) << "Mismatch at byte " << i;
    }
}

TEST_P(ze_image_memory_tests, can_copy_between_images) {
    const auto& image_layout = GetParam().image_layout;
    auto src_mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    auto dst_mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    ASSERT_NE(src_mem, nullptr);
    ASSERT_NE(dst_mem, nullptr);

    const size_t bytes = src_mem->size();
    std::vector<uint8_t> src_data(bytes);
    std::iota(src_data.begin(), src_data.end(), uint8_t{0});

    OV_ASSERT_NO_THROW(src_mem->copy_from(*ctx.ze_test_stream, src_data.data(), true));
    OV_ASSERT_NO_THROW(dst_mem->copy_from(*ctx.ze_test_stream, *src_mem, true));

    std::vector<uint8_t> dst_data(bytes, 0xFF);
    OV_ASSERT_NO_THROW(dst_mem->copy_to(*ctx.ze_test_stream, dst_data.data(), true));

    EXPECT_EQ(src_data, dst_data);
}

TEST_P(ze_image_memory_tests, can_lock_for_read) {
    const auto& image_layout = GetParam().image_layout;
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    ASSERT_NE(mem, nullptr);

    const size_t bytes = mem->size();
    std::vector<uint8_t> src(bytes);
    std::iota(src.begin(), src.end(), uint8_t{0});
    OV_ASSERT_NO_THROW(mem->copy_from(*ctx.ze_test_stream, src.data(), true));

    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = mem->lock(*ctx.ze_test_stream, mem_lock_type::read));
    ASSERT_NE(ptr, nullptr);

    for (size_t i = 0; i < bytes; i++) {
        EXPECT_EQ(static_cast<uint8_t*>(ptr)[i], src[i]) << "Mismatch at byte " << i;
    }

    OV_ASSERT_NO_THROW(mem->unlock(*ctx.ze_test_stream));
}

TEST_P(ze_image_memory_tests, can_lock_for_write) {
    const auto& image_layout = GetParam().image_layout;
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    ASSERT_NE(mem, nullptr);

    const size_t bytes = mem->size();
    std::vector<uint8_t> expected(bytes, 0xCC);

    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = mem->lock(*ctx.ze_test_stream, mem_lock_type::write));
    ASSERT_NE(ptr, nullptr);
    std::fill(static_cast<uint8_t*>(ptr), static_cast<uint8_t*>(ptr) + bytes, uint8_t{0xCC});
    OV_ASSERT_NO_THROW(mem->unlock(*ctx.ze_test_stream));

    std::vector<uint8_t> readback(bytes, 0x00);
    OV_ASSERT_NO_THROW(mem->copy_to(*ctx.ze_test_stream, readback.data(), true));

    EXPECT_EQ(readback, expected);
}

TEST_P(ze_image_memory_tests, can_lock_for_read_write) {
    const auto& image_layout = GetParam().image_layout;
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image, true);
    ASSERT_NE(mem, nullptr);

    const size_t bytes = mem->size();
    std::vector<uint8_t> expected(bytes, 1);

    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = mem->lock(*ctx.ze_test_stream, mem_lock_type::read_write));
    ASSERT_NE(ptr, nullptr);
    for (size_t i = 0; i < bytes; i++) {
        static_cast<uint8_t*>(ptr)[i] += 1;
    }
    OV_ASSERT_NO_THROW(mem->unlock(*ctx.ze_test_stream));

    std::vector<uint8_t> readback(bytes, 0x00);
    OV_ASSERT_NO_THROW(mem->copy_to(*ctx.ze_test_stream, readback.data(), true));

    EXPECT_EQ(readback, expected);
}

INSTANTIATE_TEST_SUITE_P(gpu_ze_runtime,
    ze_image_memory_tests,
    ::testing::ValuesIn(all_image_layouts),
    ze_image_memory_tests::get_test_case_name);
