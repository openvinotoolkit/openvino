// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "intel_gpu/runtime/device.hpp"
#include "runtime/ocl/ocl_device_detector.hpp"
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

    void set_mem_caps(memory_capabilities memory_capabilities) override {
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
