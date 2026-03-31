// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#include "test_utils.h"

#include "intel_gpu/runtime/memory.hpp"
#include "ze/ze_memory.hpp"

#include <exception>
#include <memory>
#include <vector>

using namespace cldnn;
using namespace ::tests;


TEST(ze_engine, memory_creation) {
	std::shared_ptr<engine> ze_test_engine = nullptr;
	try {
		ze_test_engine = create_test_engine(engine_types::ze, runtime_types::ze);
	} catch (const std::exception& e) {
		GTEST_SKIP() << e.what();
	}

	ASSERT_NE(ze_test_engine, nullptr);
	ASSERT_EQ(ze_test_engine->type(), engine_types::ze);
	ASSERT_EQ(ze_test_engine->runtime_type(), runtime_types::ze);

	auto ze_test_stream = ze_test_engine->create_stream(get_test_default_config(*ze_test_engine));

	std::shared_ptr<memory> mem = nullptr;
	layout layout_to_allocate = {{2, 4}, data_types::u8, format::bfyx};

	OV_ASSERT_NO_THROW(mem = ze_test_engine->allocate_memory(layout_to_allocate));
	ASSERT_NE(mem, nullptr);
	ASSERT_EQ(mem->get_layout(), layout_to_allocate);
	ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
	ASSERT_TRUE(mem->is_allocated_by(*ze_test_engine));

	if (ze_test_engine->supports_allocation(allocation_type::usm_host)) {
		OV_ASSERT_NO_THROW(mem = ze_test_engine->allocate_memory(layout_to_allocate, allocation_type::usm_host));
		ASSERT_NE(mem, nullptr);
		ASSERT_EQ(mem->get_layout(), layout_to_allocate);
		ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_host);
		ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
		ASSERT_TRUE(mem->is_allocated_by(*ze_test_engine));
		ASSERT_EQ(ze_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_host);
	}

	if (ze_test_engine->supports_allocation(allocation_type::usm_device)) {
		OV_ASSERT_NO_THROW(mem = ze_test_engine->allocate_memory(layout_to_allocate, allocation_type::usm_device));
		ASSERT_NE(mem, nullptr);
		ASSERT_EQ(mem->get_layout(), layout_to_allocate);
		ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_device);
		ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
		ASSERT_TRUE(mem->is_allocated_by(*ze_test_engine));
		ASSERT_EQ(ze_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_device);
	}

	if (ze_test_engine->supports_allocation(allocation_type::usm_shared)) {
		OV_ASSERT_NO_THROW(mem = ze_test_engine->allocate_memory(layout_to_allocate, allocation_type::usm_shared));
		ASSERT_NE(mem, nullptr);
		ASSERT_EQ(mem->get_layout(), layout_to_allocate);
		ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_shared);
		ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
		ASSERT_TRUE(mem->is_allocated_by(*ze_test_engine));
		ASSERT_EQ(ze_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_shared);
	}

	std::vector<uint8_t> host_data(2 * 4);
	OV_ASSERT_NO_THROW(mem = ze_test_engine->attach_memory(layout_to_allocate, host_data.data()));
	ASSERT_NE(mem, nullptr);
	ASSERT_EQ(mem->get_layout(), layout_to_allocate);
	ASSERT_NE(std::dynamic_pointer_cast<simple_attached_memory>(mem), nullptr);
	ASSERT_FALSE(mem->is_allocated_by(*ze_test_engine));
	ASSERT_EQ(std::dynamic_pointer_cast<simple_attached_memory>(mem)->lock(*ze_test_stream, mem_lock_type::read), host_data.data());
}

TEST(ze_engine, memory_creation_usm_host) {
    std::shared_ptr<engine> ze_test_engine = nullptr;
    try {
        ze_test_engine = create_test_engine(engine_types::ze, runtime_types::ze);
    } catch (const std::exception& e) {
        GTEST_SKIP() << e.what();
    }

    ASSERT_NE(ze_test_engine, nullptr);

    if (!ze_test_engine->supports_allocation(allocation_type::usm_host)) {
        GTEST_SKIP() << "usm_host allocation is not supported on this device";
    }

    auto ze_test_stream = ze_test_engine->create_stream(get_test_default_config(*ze_test_engine));

    const layout test_layout = {{1, 1, 16, 1}, data_types::u8, format::bfyx};

    memory::ptr mem = nullptr;
    OV_ASSERT_NO_THROW(mem = ze_test_engine->allocate_memory(test_layout, allocation_type::usm_host));

    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), test_layout);
    ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_host);
    ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(*ze_test_engine));
    ASSERT_EQ(ze_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_host);

    std::vector<uint8_t> src(16, 7);
    std::vector<uint8_t> dst(16, 0);

    OV_ASSERT_NO_THROW(mem->copy_from(*ze_test_stream, src.data(), true));
    OV_ASSERT_NO_THROW(mem->copy_to(*ze_test_stream, dst.data(), true));
    ASSERT_EQ(src, dst);
}

TEST(ze_engine, memory_creation_usm_device) {
    std::shared_ptr<engine> ze_test_engine = nullptr;
    try {
        ze_test_engine = create_test_engine(engine_types::ze, runtime_types::ze);
    } catch (const std::exception& e) {
        GTEST_SKIP() << e.what();
    }

    ASSERT_NE(ze_test_engine, nullptr);

    if (!ze_test_engine->supports_allocation(allocation_type::usm_device)) {
        GTEST_SKIP() << "usm_device allocation is not supported on this device";
    }

    auto ze_test_stream = ze_test_engine->create_stream(get_test_default_config(*ze_test_engine));

    const layout test_layout = {{1, 1, 16, 1}, data_types::u8, format::bfyx};

    memory::ptr mem = nullptr;
    OV_ASSERT_NO_THROW(mem = ze_test_engine->allocate_memory(test_layout, allocation_type::usm_device));

    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), test_layout);
    ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_device);
    ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(*ze_test_engine));
    ASSERT_EQ(ze_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_device);

    std::vector<uint8_t> src(16, 7);
    std::vector<uint8_t> dst(16, 0);

    OV_ASSERT_NO_THROW(mem->copy_from(*ze_test_stream, src.data(), true));
    OV_ASSERT_NO_THROW(mem->copy_to(*ze_test_stream, dst.data(), true));
    ASSERT_EQ(src, dst);
}

TEST(ze_engine, memory_creation_usm_shared) {
    std::shared_ptr<engine> ze_test_engine = nullptr;
    try {
        ze_test_engine = create_test_engine(engine_types::ze, runtime_types::ze);
    } catch (const std::exception& e) {
        GTEST_SKIP() << e.what();
    }

    ASSERT_NE(ze_test_engine, nullptr);

    if (!ze_test_engine->supports_allocation(allocation_type::usm_shared)) {
        GTEST_SKIP() << "usm_shared allocation is not supported on this device";
    }

    auto ze_test_stream = ze_test_engine->create_stream(get_test_default_config(*ze_test_engine));

    const layout test_layout = {{1, 1, 16, 1}, data_types::u8, format::bfyx};

    memory::ptr mem = nullptr;
    OV_ASSERT_NO_THROW(mem = ze_test_engine->allocate_memory(test_layout, allocation_type::usm_shared));

    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), test_layout);
    ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_shared);
    ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(*ze_test_engine));
    ASSERT_EQ(ze_test_engine->detect_usm_allocation_type(mem->buffer_ptr()), allocation_type::usm_shared);

    std::vector<uint8_t> src(16, 7);
    std::vector<uint8_t> dst(16, 0);

    OV_ASSERT_NO_THROW(mem->copy_from(*ze_test_stream, src.data(), true));
    OV_ASSERT_NO_THROW(mem->copy_to(*ze_test_stream, dst.data(), true));
    ASSERT_EQ(src, dst);
}

TEST(ze_engine, large_allocation) {
	// This test is used for manual testing only.
	GTEST_SKIP();

	std::shared_ptr<engine> ze_test_engine = nullptr;
	try {
		ze_test_engine = create_test_engine(engine_types::ze, runtime_types::ze);
	} catch (const std::exception& e) {
		GTEST_SKIP() << e.what();
	}

	ASSERT_NE(ze_test_engine, nullptr);

	std::shared_ptr<memory> mem = nullptr;
	ov::Shape sz_6gb = {6, 1024, 1024, 1024};
	layout layout_to_allocate = {sz_6gb, data_types::u8, format::bfyx};

	ze_test_engine->set_enable_large_allocations(true);

	if (ze_test_engine->supports_allocation(allocation_type::usm_device) &&
		ov::shape_size(sz_6gb) < ze_test_engine->get_device_info().max_global_mem_size) {
		OV_ASSERT_NO_THROW(mem = ze_test_engine->allocate_memory(layout_to_allocate, allocation_type::usm_device));
		ASSERT_NE(mem, nullptr);
		ASSERT_EQ(mem->get_layout(), layout_to_allocate);
		ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
		ASSERT_TRUE(mem->is_allocated_by(*ze_test_engine));
		ASSERT_EQ(mem->get_allocation_type(), allocation_type::usm_device);
	}
}

#endif  // OV_GPU_WITH_ZE_RT
