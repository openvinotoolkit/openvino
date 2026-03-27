// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#include "test_utils.h"

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "runtime/ze/ze_device_detector.hpp"
#include "runtime/ze/ze_device.hpp"
#include "runtime/ze/ze_event.hpp"
#include "runtime/ze/ze_base_event.hpp"
#include "runtime/ze/ze_counter_based_event.hpp"
#include "ze/ze_memory.hpp"

#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace {

struct ze_test_context {
	std::shared_ptr<engine> ze_test_engine;
	std::shared_ptr<stream> ze_test_stream;
};

ze_test_context create_ze_test_context() {
	ze_test_context ctx;
	try {
		ctx.ze_test_engine = create_test_engine(engine_types::ze, runtime_types::ze);
	} catch (const std::exception& e) {
		OPENVINO_THROW(e.what());
	}

	OPENVINO_ASSERT(ctx.ze_test_engine != nullptr, "[GPU] Failed to create ZE engine for tests");
	ctx.ze_test_stream = ctx.ze_test_engine->create_stream(get_test_default_config(*ctx.ze_test_engine));
	return ctx;
}

}  // namespace

/*
ENGINE TESTS:
*/

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

/*
DEVICE TESTS:
*/

namespace {

struct dummy_device : public device {
public:
	dummy_device(uint32_t vendor_id, device_type type, size_t device_id) : _mem_caps({}) {
		_info = device_info{};
		_info.vendor_id = vendor_id;
		_info.dev_type = type;
		_info.device_id = static_cast<uint32_t>(device_id);
	}

	const device_info& get_info() const override { return _info; }
	memory_capabilities get_mem_caps() const override { return _mem_caps; }
	bool is_same(const device::ptr other) override {
		return this == other.get();
	}

	void initialize() override {}

	bool is_initialized() const override {
		return true;
	}

	void set_mem_caps(const memory_capabilities& memory_capabilities) override {
		_mem_caps = memory_capabilities;
	}
	~dummy_device() = default;

private:
	device_info _info;
	memory_capabilities _mem_caps;
};

}  // namespace

TEST(ze_devices_test, sort_order_single_vendor) {
	size_t device_id = 0;
	std::vector<device::ptr> devices_list;
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::integrated_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));

	auto sorted_list = sort_devices(devices_list);

	std::vector<size_t> expected_devices_order = {2, 0, 1, 3, 4};

	std::vector<size_t> actual_devices_order;
	std::transform(sorted_list.begin(), sorted_list.end(), std::back_inserter(actual_devices_order), [](const device::ptr& d) -> size_t {
		return d->get_info().device_id;
	});

	ASSERT_EQ(expected_devices_order, actual_devices_order);
}

TEST(ze_devices_test, sort_order_two_vendors) {
	size_t device_id = 0;
	const auto OTHER_VENDOR_ID = 0x123;
	std::vector<device::ptr> devices_list;
	devices_list.push_back(std::make_shared<dummy_device>(OTHER_VENDOR_ID, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(OTHER_VENDOR_ID, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::integrated_gpu, device_id++));

	auto sorted_list = sort_devices(devices_list);

	std::vector<size_t> expected_devices_order = {3, 2, 0, 1};

	std::vector<size_t> actual_devices_order;
	std::transform(sorted_list.begin(), sorted_list.end(), std::back_inserter(actual_devices_order), [](const device::ptr& d) -> size_t {
		return d->get_info().device_id;
	});

	ASSERT_EQ(expected_devices_order, actual_devices_order);
}

TEST(ze_devices_test, sort_order_three_vendors) {
	size_t device_id = 0;
	const auto OTHER_VENDOR_ID1 = 0x123;
	const auto OTHER_VENDOR_ID2 = 0x1234;
	std::vector<device::ptr> devices_list;
	devices_list.push_back(std::make_shared<dummy_device>(OTHER_VENDOR_ID1, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(OTHER_VENDOR_ID1, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::integrated_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(OTHER_VENDOR_ID2, device_type::discrete_gpu, device_id++));
	devices_list.push_back(std::make_shared<dummy_device>(OTHER_VENDOR_ID2, device_type::discrete_gpu, device_id++));

	auto sorted_list = sort_devices(devices_list);

	std::vector<size_t> expected_devices_order = {2, 3, 0, 1, 4, 5};

	std::vector<size_t> actual_devices_order;
	std::transform(sorted_list.begin(), sorted_list.end(), std::back_inserter(actual_devices_order), [](const device::ptr& d) -> size_t {
		return d->get_info().device_id;
	});

	ASSERT_EQ(expected_devices_order, actual_devices_order);
}

TEST(ze_devices_test, is_same_device) {
	ze::ze_device_detector device_detector;

	const bool initialize = false;
	auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, std::numeric_limits<int>::max() /* ignore sub-devices */, initialize);

	if (devices.empty())
		GTEST_SKIP() << "No available devices found";

	for (const auto& device : devices) {
		ASSERT_TRUE(device.second->is_same(device.second));
	}

	if (devices.size() > 1) {
		auto first_device = devices.begin()->second;
		auto second_device = devices.rbegin()->second;

		ASSERT_FALSE(first_device->is_same(second_device));
	}
}

TEST(ze_devices_test, on_demand_initialization) {
	ze::ze_device_detector device_detector;

	const bool initialize = false;
	auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, std::numeric_limits<int>::max() /* ignore sub-devices */, initialize);

	if (devices.empty())
		GTEST_SKIP() << "No available devices found";

	for (const auto& device : devices) {
		auto ze_device = std::dynamic_pointer_cast<ze::ze_device>(device.second);
		ASSERT_NE(ze_device, nullptr);
		ASSERT_FALSE(ze_device->is_initialized());
		ASSERT_EQ(ze_device->get_context(), nullptr);
		ASSERT_TRUE(ze_device->get_device() != nullptr);
		ASSERT_TRUE(ze_device->get_driver() != nullptr);
		ASSERT_FALSE(ze_device->get_info().execution_units_count == 0);
		ASSERT_FALSE(ze_device->get_info().vendor_id == 0);
	}

	for (const auto& device : devices) {
		ASSERT_NO_THROW(device.second->initialize());
	}

	for (const auto& device : devices) {
		auto ze_device = std::dynamic_pointer_cast<ze::ze_device>(device.second);
		ASSERT_NE(ze_device, nullptr);
		ASSERT_TRUE(ze_device->is_initialized());
		ASSERT_TRUE(ze_device->get_device() != nullptr);
		ASSERT_TRUE(ze_device->get_context() != nullptr);
	}
}

TEST(ze_devices_test, user_context_initialization_not_implemented) {
	ze::ze_device_detector device_detector;

	const bool initialize = true;
	auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, -1, initialize);

	if (devices.empty())
		GTEST_SKIP() << "No available devices found";

	auto initialized_device = std::dynamic_pointer_cast<ze::ze_device>(devices.begin()->second);
	ASSERT_NE(initialized_device, nullptr);
	auto user_context = initialized_device->get_context();
	ASSERT_TRUE(user_context != nullptr);

	ASSERT_ANY_THROW(device_detector.get_available_devices(user_context, nullptr, 0, std::numeric_limits<int>::max() /* ignore sub-devices */));
}

TEST(ze_devices_test, user_device_initialization_not_implemented) {
	ze::ze_device_detector device_detector;

	const bool initialize = true;
	auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, -1, initialize);

	if (devices.empty())
		GTEST_SKIP() << "No available devices found";

	auto initialized_device = std::dynamic_pointer_cast<ze::ze_device>(devices.begin()->second);
	ASSERT_NE(initialized_device, nullptr);
	auto user_device = initialized_device->get_device();
	ASSERT_TRUE(user_device != nullptr);

	ASSERT_ANY_THROW(device_detector.get_available_devices(nullptr, user_device, 0, std::numeric_limits<int>::max() /* ignore sub-devices */));
}

/*
EVENTS TESTS:
*/

// user events:

TEST(ze_event, can_create_user_event_as_complete) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(true);

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_TRUE(user_ev->is_set());
}

TEST(ze_event, can_create_user_event_as_not_complete) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(false);

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_FALSE(user_ev->is_set());
}

TEST(ze_event, can_create_user_event_as_not_complete_and_set) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(false);

	
	if (std::dynamic_pointer_cast<ze::ze_counter_based_event>(user_ev) != nullptr) {
		GTEST_SKIP() << "Counter based events are always created as complete";
	}

	user_ev->set();
	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_TRUE(user_ev->is_set());
}

TEST(ze_event, can_create_user_event_as_complete_and_wait) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(true);
	user_ev->wait();

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_TRUE(user_ev->is_set());
}

TEST(ze_event, can_create_user_event_as_not_complete_set_and_wait) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(false);
	user_ev->set();
	user_ev->wait();

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_TRUE(user_ev->is_set());
}

// counter based events:

TEST(ze_event, can_create_counter_based_event) {
	auto ctx = create_ze_test_context();
	auto base_ev = ctx.ze_test_stream->create_base_event();

	if (std::dynamic_pointer_cast<ze::ze_counter_based_event>(base_ev) == nullptr)
		GTEST_SKIP() << "Counter based events not supported by this stream";

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(base_ev), nullptr);
	ASSERT_TRUE(base_ev->is_set());
}

TEST(ze_event, can_create_counter_based_event_and_wait) {
	auto ctx = create_ze_test_context();
	auto base_ev = ctx.ze_test_stream->create_base_event();

	if (std::dynamic_pointer_cast<ze::ze_counter_based_event>(base_ev) == nullptr)
		GTEST_SKIP() << "Counter based events not supported by this stream";

	base_ev->wait();

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(base_ev), nullptr);
	ASSERT_TRUE(base_ev->is_set());
}

/*
USM MEMORY TESTS:
*/

namespace {
	
std::vector<allocation_type> get_supported_ze_alloc_types(const std::shared_ptr<engine>& ze_test_engine) {
	std::vector<allocation_type> alloc_types;
	if (ze_test_engine->supports_allocation(allocation_type::usm_host)) {
		alloc_types.push_back(allocation_type::usm_host);
	}
	if (ze_test_engine->supports_allocation(allocation_type::usm_shared)) {
		alloc_types.push_back(allocation_type::usm_shared);
	}
	if (ze_test_engine->supports_allocation(allocation_type::usm_device)) {
		alloc_types.push_back(allocation_type::usm_device);
	}

	return alloc_types;
}

} // namespace

TEST(ze_usm_memory, copy_and_read_buffer) {
	auto ctx = create_ze_test_context();
	auto alloc_types = get_supported_ze_alloc_types(ctx.ze_test_engine);
	if (alloc_types.empty()) {
		GTEST_SKIP() << "No supported ZE USM allocation types";
	}

	const size_t values_count = 100;
	const layout linear_layout = {{1, 1, static_cast<int32_t>(values_count), 1}, data_types::f32, format::bfyx};
	std::vector<float> src_buffer(values_count);
	std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

	for (auto alloc_type : alloc_types) {
		SCOPED_TRACE(static_cast<int>(alloc_type));
		auto mem = ctx.ze_test_engine->allocate_memory(linear_layout, alloc_type);
		ASSERT_NE(mem, nullptr);

		OV_ASSERT_NO_THROW(mem->copy_from(*ctx.ze_test_stream, src_buffer.data(), true));

		std::vector<float> dst_buffer(values_count, -1.0f);
		OV_ASSERT_NO_THROW(mem->copy_to(*ctx.ze_test_stream, dst_buffer.data(), true));
		ASSERT_EQ(src_buffer, dst_buffer);
	}
}

TEST(ze_usm_memory, offset_copy_between_memories) {
	auto ctx = create_ze_test_context();
	auto alloc_types = get_supported_ze_alloc_types(ctx.ze_test_engine);
	if (alloc_types.empty()) {
		GTEST_SKIP() << "No supported ZE USM allocation types";
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

			auto src_mem = ctx.ze_test_engine->allocate_memory(src_layout, src_alloc);
			auto dst_mem = ctx.ze_test_engine->allocate_memory(dst_layout, dst_alloc);
			ASSERT_NE(src_mem, nullptr);
			ASSERT_NE(dst_mem, nullptr);

			OV_ASSERT_NO_THROW(src_mem->copy_from(*ctx.ze_test_stream, src_data.data(), 0, 0, src_size, true));
			OV_ASSERT_NO_THROW(dst_mem->copy_from(*ctx.ze_test_stream, *src_mem, src_offset, dst_offset, copy_size, true));

			std::vector<uint8_t> actual(copy_size, 0);
			OV_ASSERT_NO_THROW(dst_mem->copy_to(*ctx.ze_test_stream, actual.data(), dst_offset, 0, copy_size, true));

			for (size_t i = 0; i < copy_size; i++) {
				ASSERT_EQ(src_data[i + src_offset], actual[i]);
			}
		}
	}
}

TEST(ze_usm_memory, offset_copy_host_roundtrip) {
	auto ctx = create_ze_test_context();
	auto alloc_types = get_supported_ze_alloc_types(ctx.ze_test_engine);
	if (alloc_types.empty()) {
		GTEST_SKIP() << "No supported ZE USM allocation types";
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
		auto mem = ctx.ze_test_engine->allocate_memory(mem_layout, alloc_type);
		ASSERT_NE(mem, nullptr);

		OV_ASSERT_NO_THROW(mem->copy_from(*ctx.ze_test_stream, src_buffer.data(), src_offset, dst_offset, copy_size, true));
		OV_ASSERT_NO_THROW(mem->copy_to(*ctx.ze_test_stream, dst_buffer.data(), dst_offset, src_offset, copy_size, true));

		for (size_t i = 0; i < copy_size; i++) {
			ASSERT_EQ(src_buffer[src_offset + i], dst_buffer[src_offset + i]);
		}
	}
}

TEST(ze_usm_memory, allocate_different_types) {
	auto ctx = create_ze_test_context();
	auto alloc_types = get_supported_ze_alloc_types(ctx.ze_test_engine);
	if (alloc_types.empty()) {
		GTEST_SKIP() << "No supported ZE USM allocation types";
	}

	const layout test_layout = {{100}, data_types::f32, format::bfyx};
	for (auto alloc_type : alloc_types) {
		SCOPED_TRACE(static_cast<int>(alloc_type));
		auto mem = ctx.ze_test_engine->allocate_memory(test_layout, alloc_type);
		ASSERT_NE(mem, nullptr);
		ASSERT_EQ(mem->get_allocation_type(), alloc_type);
		ASSERT_NE(std::dynamic_pointer_cast<ze::gpu_usm>(mem), nullptr);
	}
}

TEST(ze_usm_memory, fill_buffer) {
	auto ctx = create_ze_test_context();
	auto alloc_types = get_supported_ze_alloc_types(ctx.ze_test_engine);
	if (alloc_types.empty()) {
		GTEST_SKIP() << "No supported ZE USM allocation types";
	}

	const size_t values_count = 100;
	const layout linear_layout = {{1, 1, static_cast<int32_t>(values_count), 1}, data_types::u8, format::bfyx};
	const unsigned char fill_pattern = 42;

	for (auto alloc_type : alloc_types) {
		SCOPED_TRACE(static_cast<int>(alloc_type));
		auto mem = ctx.ze_test_engine->allocate_memory(linear_layout, alloc_type);
		ASSERT_NE(mem, nullptr);

		OV_ASSERT_NO_THROW(mem->fill(*ctx.ze_test_stream, fill_pattern, {}, true));

		if (alloc_type == allocation_type::usm_host || alloc_type == allocation_type::usm_shared) {
			auto ptr = std::dynamic_pointer_cast<ze::gpu_usm>(mem)->buffer_ptr();
			for (size_t i = 0; i < values_count; i++) {
				ASSERT_EQ(*(static_cast<unsigned char*>(ptr) + i), fill_pattern);
			}
		} else if (alloc_type == allocation_type::usm_device) {
			std::vector<uint8_t> host_buffer(values_count);
			OV_ASSERT_NO_THROW(mem->copy_to(*ctx.ze_test_stream, host_buffer.data(), true));
			for (size_t i = 0; i < values_count; i++) {
				ASSERT_EQ(host_buffer[i], fill_pattern);
			}
		}
	}
}

TEST(ze_usm_memory, copy_between_different_alloc_types) {
	auto ctx = create_ze_test_context();
	auto alloc_types = get_supported_ze_alloc_types(ctx.ze_test_engine);
	if (alloc_types.size() < 2) {
		GTEST_SKIP() << "Need at least 2 supported ZE USB allocation types";
	}

	const size_t values_count = 100;
	const layout linear_layout = {{1, 1, static_cast<int32_t>(values_count), 1}, data_types::f32, format::bfyx};
	std::vector<float> src_buffer(values_count);
	std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

	auto src_mem = ctx.ze_test_engine->allocate_memory(linear_layout, alloc_types[0]);
	OV_ASSERT_NO_THROW(src_mem->copy_from(*ctx.ze_test_stream, src_buffer.data(), true));

	for (size_t i = 1; i < alloc_types.size(); i++) {
		SCOPED_TRACE(static_cast<int>(alloc_types[i]));
		auto dst_mem = ctx.ze_test_engine->allocate_memory(linear_layout, alloc_types[i]);
		ASSERT_NE(dst_mem, nullptr);

		OV_ASSERT_NO_THROW(dst_mem->copy_from(*ctx.ze_test_stream, *src_mem, true));

		std::vector<float> result(values_count);
		OV_ASSERT_NO_THROW(dst_mem->copy_to(*ctx.ze_test_stream, result.data(), true));
		ASSERT_EQ(src_buffer, result);
	}
}

TEST(ze_usm_memory, copy_small_to_large_buffer) {
	auto ctx = create_ze_test_context();
	auto alloc_types = get_supported_ze_alloc_types(ctx.ze_test_engine);
	if (alloc_types.empty()) {
		GTEST_SKIP() << "No supported ZE USM allocation types";
	}

	const size_t small_size = 512;
	const size_t large_size = 768;

	const layout small_layout = {{static_cast<int64_t>(small_size)}, data_types::u8, format::bfyx};
	const layout large_layout = {{static_cast<int64_t>(large_size)}, data_types::u8, format::bfyx};

	auto small_mem = ctx.ze_test_engine->allocate_memory(small_layout, alloc_types[0]);
	auto large_mem = ctx.ze_test_engine->allocate_memory(large_layout, alloc_types[0]);
	ASSERT_NE(small_mem, nullptr);
	ASSERT_NE(large_mem, nullptr);

	OV_ASSERT_NO_THROW(small_mem->copy_to(*ctx.ze_test_stream, *large_mem, true));
}


#endif  // OV_GPU_WITH_ZE_RT
