// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "test_utils.h"

#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(engine, memory_creation) {
    auto& engine = get_test_engine();

    std::shared_ptr<memory> mem = nullptr;
    layout layout_to_allocate = {{2, 4}, data_types::u8, format::bfyx};
    OV_ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate));
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), layout_to_allocate);
    ASSERT_TRUE(mem->is_allocated_by(engine));

    OV_ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate, allocation_type::cl_mem));
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), layout_to_allocate);
    ASSERT_NE(std::dynamic_pointer_cast<ocl::gpu_buffer>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(engine));

    if (engine.supports_allocation(allocation_type::usm_host)) {
        OV_ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate, allocation_type::usm_host));
        ASSERT_NE(mem, nullptr);
        ASSERT_EQ(mem->get_layout(), layout_to_allocate);
        ASSERT_NE(std::dynamic_pointer_cast<ocl::gpu_usm>(mem), nullptr);
        ASSERT_TRUE(mem->is_allocated_by(engine));
    }

    if (engine.supports_allocation(allocation_type::usm_device)) {
        OV_ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate, allocation_type::usm_device));
        ASSERT_NE(mem, nullptr);
        ASSERT_EQ(mem->get_layout(), layout_to_allocate);
        ASSERT_NE(std::dynamic_pointer_cast<ocl::gpu_usm>(mem), nullptr);
        ASSERT_TRUE(mem->is_allocated_by(engine));
    }

    std::vector<uint8_t> host_data(2*4);
    OV_ASSERT_NO_THROW(mem = engine.attach_memory(layout_to_allocate, host_data.data()));
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), layout_to_allocate);
    ASSERT_NE(std::dynamic_pointer_cast<simple_attached_memory>(mem), nullptr);
    ASSERT_FALSE(mem->is_allocated_by(engine));
    ASSERT_EQ(std::dynamic_pointer_cast<simple_attached_memory>(mem)->lock(get_test_stream(), mem_lock_type::read), host_data.data());
}

TEST(engine, large_allocation) {
    // This test is used for manual testing only.
    GTEST_SKIP();

    auto& engine = get_test_engine();

    std::shared_ptr<memory> mem = nullptr;
    ov::Shape sz_6gb = {6, 1024, 1024, 1024};
    layout layout_to_allocate = {sz_6gb, data_types::u8, format::bfyx};

    engine.set_enable_large_allocations(true);

    if (engine.supports_allocation(allocation_type::usm_device) && ov::shape_size(sz_6gb) < engine.get_device_info().max_global_mem_size) {
        OV_ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate, allocation_type::usm_host));
        ASSERT_NE(mem, nullptr);
        ASSERT_EQ(mem->get_layout(), layout_to_allocate);
        ASSERT_NE(std::dynamic_pointer_cast<ocl::gpu_usm>(mem), nullptr);
        ASSERT_TRUE(mem->is_allocated_by(engine));
    }
}
