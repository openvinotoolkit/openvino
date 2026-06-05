// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#ifdef OV_GPU_WITH_SYCL_RT

#include "sycl_test_context.hpp"

#include "intel_gpu/runtime/memory.hpp"
#include "runtime/sycl/sycl_memory.hpp"

#include <memory>
#include <vector>

using namespace cldnn;
using namespace sycl_tests;

TEST(sycl_engine, memory_creation) {
    auto sycl_test_context = create_sycl_test_context();
    auto sycl_test_engine = sycl_test_context.sycl_test_engine;
    auto sycl_test_stream = sycl_test_context.sycl_test_stream;

    ASSERT_NE(sycl_test_engine, nullptr);
    ASSERT_EQ(sycl_test_engine->type(), engine_types::sycl);
    ASSERT_EQ(sycl_test_engine->runtime_type(), runtime_types::sycl);

    std::shared_ptr<memory> mem = nullptr;
    layout layout_to_allocate = {{2, 4}, data_types::u8, format::bfyx};

    OV_ASSERT_NO_THROW(mem = sycl_test_engine->allocate_memory(layout_to_allocate));
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), layout_to_allocate);
    ASSERT_TRUE(mem->is_allocated_by(*sycl_test_engine));

    std::vector<uint8_t> host_data(2 * 4);
    OV_ASSERT_NO_THROW(mem = sycl_test_engine->attach_memory(layout_to_allocate, host_data.data()));
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), layout_to_allocate);
    ASSERT_NE(std::dynamic_pointer_cast<simple_attached_memory>(mem), nullptr);
    ASSERT_FALSE(mem->is_allocated_by(*sycl_test_engine));
    ASSERT_EQ(std::dynamic_pointer_cast<simple_attached_memory>(mem)->lock(*sycl_test_stream, mem_lock_type::read), host_data.data());
}

TEST(sycl_engine, memory_creation_sycl_buffer) {
    auto sycl_test_context = create_sycl_test_context();
    auto sycl_test_engine = sycl_test_context.sycl_test_engine;
    auto sycl_test_stream = sycl_test_context.sycl_test_stream;

    ASSERT_NE(sycl_test_engine, nullptr);

    if (!sycl_test_engine->supports_allocation(allocation_type::sycl_buffer)) {
        GTEST_SKIP() << "sycl_buffer allocation is not supported on this device";
    }

    const layout test_layout = {{1, 1, 16, 1}, data_types::u8, format::bfyx};

    memory::ptr mem = nullptr;
    OV_ASSERT_NO_THROW(mem = sycl_test_engine->allocate_memory(test_layout, allocation_type::sycl_buffer));

    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), test_layout);
    ASSERT_EQ(mem->get_allocation_type(), allocation_type::sycl_buffer);
    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::gpu_buffer>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(*sycl_test_engine));

    std::vector<uint8_t> src(16, 7);
    std::vector<uint8_t> dst(16, 0);

    OV_ASSERT_NO_THROW(mem->copy_from(*sycl_test_stream, src.data(), true));
    OV_ASSERT_NO_THROW(mem->copy_to(*sycl_test_stream, dst.data(), true));
    ASSERT_EQ(src, dst);
}

TEST(sycl_engine, memory_creation_usm_host) {
    auto sycl_test_context = create_sycl_test_context();
    auto sycl_test_engine = sycl_test_context.sycl_test_engine;
    auto sycl_test_stream = sycl_test_context.sycl_test_stream;

    ASSERT_NE(sycl_test_engine, nullptr);

    if (!sycl_test_engine->supports_allocation(allocation_type::usm_host)) {
        GTEST_SKIP() << "usm_host allocation is not supported on this device";
    }

    const layout test_layout = {{1, 1, 16, 1}, data_types::u8, format::bfyx};

    memory::ptr mem = nullptr;
    OV_ASSERT_NO_THROW(mem = sycl_test_engine->allocate_memory(test_layout, allocation_type::usm_host));

    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), test_layout);
    ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_host);
    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::gpu_usm>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(*sycl_test_engine));
    ASSERT_EQ(sycl_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_host);

    std::vector<uint8_t> src(16, 7);
    std::vector<uint8_t> dst(16, 0);

    OV_ASSERT_NO_THROW(mem->copy_from(*sycl_test_stream, src.data(), true));
    OV_ASSERT_NO_THROW(mem->copy_to(*sycl_test_stream, dst.data(), true));
    ASSERT_EQ(src, dst);
}

TEST(sycl_engine, memory_creation_usm_device) {
    auto sycl_test_context = create_sycl_test_context();
    auto sycl_test_engine = sycl_test_context.sycl_test_engine;
    auto sycl_test_stream = sycl_test_context.sycl_test_stream;

    ASSERT_NE(sycl_test_engine, nullptr);

    if (!sycl_test_engine->supports_allocation(allocation_type::usm_device)) {
        GTEST_SKIP() << "usm_device allocation is not supported on this device";
    }

    const layout test_layout = {{1, 1, 16, 1}, data_types::u8, format::bfyx};

    memory::ptr mem = nullptr;
    OV_ASSERT_NO_THROW(mem = sycl_test_engine->allocate_memory(test_layout, allocation_type::usm_device));

    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), test_layout);
    ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_device);
    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::gpu_usm>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(*sycl_test_engine));
    ASSERT_EQ(sycl_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_device);

    std::vector<uint8_t> src(16, 7);
    std::vector<uint8_t> dst(16, 0);

    OV_ASSERT_NO_THROW(mem->copy_from(*sycl_test_stream, src.data(), true));
    OV_ASSERT_NO_THROW(mem->copy_to(*sycl_test_stream, dst.data(), true));
    ASSERT_EQ(src, dst);
}

TEST(sycl_engine, memory_creation_usm_shared) {
    auto sycl_test_context = create_sycl_test_context();
    auto sycl_test_engine = sycl_test_context.sycl_test_engine;
    auto sycl_test_stream = sycl_test_context.sycl_test_stream;

    ASSERT_NE(sycl_test_engine, nullptr);

    if (!sycl_test_engine->supports_allocation(allocation_type::usm_shared)) {
        GTEST_SKIP() << "usm_shared allocation is not supported on this device";
    }

    const layout test_layout = {{1, 1, 16, 1}, data_types::u8, format::bfyx};

    memory::ptr mem = nullptr;
    OV_ASSERT_NO_THROW(mem = sycl_test_engine->allocate_memory(test_layout, allocation_type::usm_shared));

    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), test_layout);
    ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_shared);
    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::gpu_usm>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(*sycl_test_engine));
    ASSERT_EQ(sycl_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_shared);

    std::vector<uint8_t> src(16, 7);
    std::vector<uint8_t> dst(16, 0);

    OV_ASSERT_NO_THROW(mem->copy_from(*sycl_test_stream, src.data(), true));
    OV_ASSERT_NO_THROW(mem->copy_to(*sycl_test_stream, dst.data(), true));
    ASSERT_EQ(src, dst);
}

// TODO: add the below test when usm allocation with large size is supported
// TEST(sycl_engine, large_allocation) { ... }

#endif  // OV_GPU_WITH_SYCL_RT
