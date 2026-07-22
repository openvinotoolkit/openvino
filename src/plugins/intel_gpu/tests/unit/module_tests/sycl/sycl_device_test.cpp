// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#ifdef OV_GPU_WITH_SYCL_RT

#include "sycl_test_context.hpp"

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "runtime/sycl/sycl_device_detector.hpp"
#include "runtime/sycl/sycl_device.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

using namespace cldnn;
using namespace sycl_tests;


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

TEST(sycl_devices_test, sort_order_single_vendor) {
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

TEST(sycl_devices_test, sort_order_two_vendors) {
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

TEST(sycl_devices_test, sort_order_three_vendors) {
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

TEST(sycl_devices_test, is_same_device) {
    cldnn::sycl::sycl_device_detector device_detector;

    auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, std::numeric_limits<int>::max() /* ignore sub-devices */);

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

// TODO: add test for on-demand initialization when it's implemented
// TEST(sycl_devices_test, on_demand_initialization)

TEST(sycl_devices_test, user_context_initialization) {
    cldnn::sycl::sycl_device_detector device_detector;
    auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, -1);

    if (devices.empty())
        GTEST_SKIP() << "No available devices found";

    auto initialized_device = std::dynamic_pointer_cast<cldnn::sycl::sycl_device>(devices.begin()->second);
    ASSERT_NE(initialized_device, nullptr);
    auto user_context = initialized_device->get_context();
    auto shared_devices = device_detector.get_available_devices(&user_context, nullptr, 0, std::numeric_limits<int>::max() /* ignore sub-devices */);
    ASSERT_EQ(shared_devices.size(), 1);

    auto shared_sycl_device = std::dynamic_pointer_cast<cldnn::sycl::sycl_device>(shared_devices.begin()->second);
    ASSERT_NE(shared_sycl_device, nullptr);
    ASSERT_TRUE(shared_sycl_device->is_initialized());

    ASSERT_EQ(shared_sycl_device->get_device(), initialized_device->get_device());
    ASSERT_EQ(shared_sycl_device->get_context(), initialized_device->get_context());
    ASSERT_TRUE(initialized_device->is_same(shared_sycl_device));
}

TEST(sycl_devices_test, user_device_initialization_not_implemented) {
    cldnn::sycl::sycl_device_detector device_detector;

    auto devices = device_detector.get_available_devices(nullptr, nullptr, 0, -1);

    if (devices.empty())
        GTEST_SKIP() << "No available devices found";

    auto initialized_device = std::dynamic_pointer_cast<cldnn::sycl::sycl_device>(devices.begin()->second);
    ASSERT_NE(initialized_device, nullptr);
    auto user_device = initialized_device->get_device();

    ASSERT_ANY_THROW(device_detector.get_available_devices(nullptr, &user_device, 0, std::numeric_limits<int>::max() /* ignore sub-devices */));
}

#endif  // OV_GPU_WITH_SYCL_RT
