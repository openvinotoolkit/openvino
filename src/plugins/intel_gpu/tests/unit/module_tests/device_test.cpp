// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/execution_config.hpp"
#include "module_tests/config_gpu.hpp"
#include "openvino/runtime/properties.hpp"
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

// class Test {
// public:
//     int i;
//     constexpr Test(int i) : i(i) {}
// };

// constexpr const Test test1(1);
// constexpr const Test test2(2);

// template<Test t>
// int get_prop() {
//     static_assert(false, "FAIL");
// }

// template<template<typename, ov::PropertyMutability> class prop, typename T, ov::PropertyMutability mutability>
// T get_prop() {
//     static_assert(false, "FAIL");
// }


TEST(config_test, basic) {
    ov::intel_gpu::NewExecutionConfig cfg;
    std::cerr << cfg.to_string();

    cfg.set_user_property(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    cfg.set_property(ov::hint::inference_precision(ov::element::f32));

    std::cerr << "PROF: " << cfg.enable_profiling.value << std::endl;

    std::cerr << cfg.to_string();

    std::cerr << cfg.get_property(ov::hint::inference_precision) << std::endl;
    std::cerr << cfg.get_property(ov::hint::execution_mode) << std::endl;

//     std::cerr << get_prop<ov::hint::inference_precision>() << std::endl;
//     std::cerr << get_prop<test1>() << std::endl;
}
