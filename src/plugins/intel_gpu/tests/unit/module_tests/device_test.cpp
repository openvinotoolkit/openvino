// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "intel_gpu/runtime/device.hpp"
#include "runtime/ocl/ocl_device_detector.hpp"
#include "runtime/ocl/ocl_device.hpp"

#include <memory>

using namespace cldnn;
using namespace ::tests;

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

    void initialize() override {};

    bool is_initialized() const override {
        return true;
    };

    void set_mem_caps(const memory_capabilities& memory_capabilities) override {
        _mem_caps = memory_capabilities;
    }
    ~dummy_device() = default;

private:
    device_info _info;
    memory_capabilities _mem_caps;
};

}  // namespace

TEST(devices_test, sort_order_single_vendor) {
    size_t device_id = 0;
    std::vector<device::ptr> devices_list;
    devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
    devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
    devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::integrated_gpu, device_id++));
    devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
    devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));

    auto sorted_list = ocl::ocl_device_detector::sort_devices(devices_list);

    std::vector<size_t> expected_devices_order = {2, 0, 1, 3, 4};

    std::vector<size_t> actual_devices_order;
    std::transform(sorted_list.begin(), sorted_list.end(), std::back_inserter(actual_devices_order), [](const device::ptr& d) -> size_t {
        return d->get_info().device_id;
    });

    ASSERT_EQ(expected_devices_order, actual_devices_order);
}

TEST(devices_test, sort_order_two_vendors) {
    size_t device_id = 0;
    const auto OTHER_VENDOR_ID = 0x123;
    std::vector<device::ptr> devices_list;
    devices_list.push_back(std::make_shared<dummy_device>(OTHER_VENDOR_ID, device_type::discrete_gpu, device_id++));
    devices_list.push_back(std::make_shared<dummy_device>(OTHER_VENDOR_ID, device_type::discrete_gpu, device_id++));
    devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::discrete_gpu, device_id++));
    devices_list.push_back(std::make_shared<dummy_device>(INTEL_VENDOR_ID, device_type::integrated_gpu, device_id++));

    auto sorted_list = ocl::ocl_device_detector::sort_devices(devices_list);

    std::vector<size_t> expected_devices_order = {3, 2, 0, 1};

    std::vector<size_t> actual_devices_order;
    std::transform(sorted_list.begin(), sorted_list.end(), std::back_inserter(actual_devices_order), [](const device::ptr& d) -> size_t {
        return d->get_info().device_id;
    });

    ASSERT_EQ(expected_devices_order, actual_devices_order);
}

TEST(devices_test, sort_order_three_vendors) {
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

    auto sorted_list = ocl::ocl_device_detector::sort_devices(devices_list);

    std::vector<size_t> expected_devices_order = {2, 3, 0, 1, 4, 5};

    std::vector<size_t> actual_devices_order;
    std::transform(sorted_list.begin(), sorted_list.end(), std::back_inserter(actual_devices_order), [](const device::ptr& d) -> size_t {
        return d->get_info().device_id;
    });

    ASSERT_EQ(expected_devices_order, actual_devices_order);
}

namespace cldnn::ocl {
struct ocl_device_extended : public ocl_device {
public:
    using ptr = std::shared_ptr<ocl_device_extended>;
    ocl_device_extended(const ocl_device::ptr other) : ocl_device(other, false) {}

    void set_device_info(const device_info& new_device_info) {
        _info = new_device_info;
    }
};
}  // namespace ocl

TEST(devices_test, is_same_device) {
    ocl::ocl_device_detector device_detector;

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

    auto orig_device = std::dynamic_pointer_cast<ocl::ocl_device>(devices.begin()->second);
    auto new_device = std::make_shared<ocl::ocl_device_extended>(orig_device);

    ASSERT_TRUE(new_device->is_same(orig_device));

    auto new_device_info = new_device->get_info();
    auto orig_uuid = new_device_info.uuid.uuid;
    new_device_info.uuid.uuid[0] += 1;
    new_device->set_device_info(new_device_info);
    ASSERT_FALSE(new_device->is_same(orig_device));
    new_device_info.uuid.uuid = orig_uuid;

    auto orig_pci_bus = new_device_info.pci_info.pci_bus;
    new_device_info.pci_info.pci_bus += 1;
    new_device->set_device_info(new_device_info);
    ASSERT_FALSE(new_device->is_same(orig_device));
    new_device_info.pci_info.pci_bus = orig_pci_bus;

    auto orig_sub_device_idx = new_device_info.sub_device_idx;
    new_device_info.sub_device_idx += 1;
    new_device->set_device_info(new_device_info);
    ASSERT_FALSE(new_device->is_same(orig_device));
    new_device_info.sub_device_idx = orig_sub_device_idx;
    new_device_info.pci_info.pci_bus = orig_pci_bus;

    auto orig_vendor = new_device_info.vendor_id;
    new_device_info.vendor_id += 1;
    new_device->set_device_info(new_device_info);
    ASSERT_FALSE(new_device->is_same(orig_device));
    new_device_info.vendor_id = orig_vendor;

    auto orig_eu_count = new_device_info.execution_units_count;
    new_device_info.execution_units_count += 1;
    new_device->set_device_info(new_device_info);
    ASSERT_FALSE(new_device->is_same(orig_device));
    new_device_info.execution_units_count = orig_eu_count;

    new_device->set_device_info(new_device_info);
    ASSERT_TRUE(new_device->is_same(orig_device));
}

TEST(devices_test, on_demand_initialization) {
    ocl::ocl_device_detector device_detector;

    const bool initialize = false;
    auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, std::numeric_limits<int>::max() /* ignore sub-devices */, initialize);

    // Check that devices have the expected initialization state and their descriptions are properly configured
    for (const auto& device : devices) {
        auto ocl_device = std::dynamic_pointer_cast<ocl::ocl_device>(device.second);
        auto should_be_initialized = ocl_device->get_info().vendor_id == cldnn::INTEL_VENDOR_ID;
        ASSERT_EQ(ocl_device->is_initialized(), should_be_initialized);
        ASSERT_EQ(ocl_device->get_device().get() != nullptr, should_be_initialized);
        ASSERT_EQ(ocl_device->get_context().get() != nullptr, should_be_initialized);
        ASSERT_FALSE(ocl_device->get_info().execution_units_count == 0);
        ASSERT_FALSE(ocl_device->get_info().vendor_id == 0);
        ASSERT_TRUE(ocl_device->get_info().sub_device_idx == std::numeric_limits<uint32_t>::max() /* root devices only */);
    }

    // Initialize all devices
    for (const auto& device : devices) {
        ASSERT_NO_THROW(device.second->initialize());
    }

    // Check that devices were initialized
    for (const auto& device : devices) {
        auto ocl_device = std::dynamic_pointer_cast<ocl::ocl_device>(device.second);
        ASSERT_TRUE(ocl_device->is_initialized());
        ASSERT_TRUE(ocl_device->get_device().get());
        ASSERT_TRUE(ocl_device->get_context().get());
    }
}

TEST(devices_test, user_context_initialization) {
    ocl::ocl_device_detector device_detector;

    const bool initialize = true;
    auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, -1, initialize);

    if (devices.empty())
        GTEST_SKIP() << "No available devices found";

    auto initialized_device = std::dynamic_pointer_cast<ocl::ocl_device>(devices.begin()->second);
    auto user_context = initialized_device->get_context();

    auto shared_devices = device_detector.get_available_devices(user_context.get(), nullptr, 0, std::numeric_limits<int>::max() /* ignore sub-devices */);
    ASSERT_EQ(shared_devices.size(), 1);

    auto shared_ocl_device = std::dynamic_pointer_cast<ocl::ocl_device>(shared_devices.begin()->second);
    ASSERT_TRUE(shared_ocl_device->is_initialized());

    ASSERT_EQ(shared_ocl_device->get_device(), initialized_device->get_device());
    ASSERT_EQ(shared_ocl_device->get_context(), initialized_device->get_context());
    ASSERT_TRUE(initialized_device->is_same(shared_ocl_device));
}
