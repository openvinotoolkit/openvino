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
#include <vector>

using namespace cldnn;
using namespace ::tests;

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


#endif  // OV_GPU_WITH_ZE_RT
