// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "opencl_helper_instance.hpp"
#include "ocl/ocl_device.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/runtime/device_query.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(cl_mem_check, check_write_access_type) {
    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());
    cldnn::device::ptr device = devices.begin()->second;
    for (auto& dev : devices) {
        if (dev.second->get_info().dev_type == device_type::discrete_gpu)
            device = dev.second;
    }

    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);
    auto stream = engine->create_stream({});

    size_t values_count = 100;
    size_t values_bytes_count = values_count * sizeof(float);
    std::vector<float> src_buffer(values_count);
    std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

    cldnn::layout linear_layout{{ov::Dimension(values_count)}, cldnn::data_types::f32, cldnn::format::bfyx};
    auto cldnn_mem_src = engine->allocate_memory(linear_layout, cldnn::allocation_type::cl_mem);
    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::write> lock(cldnn_mem_src, *stream);
        std::copy(src_buffer.begin(), src_buffer.end(), lock.data());
    }

    std::vector<float> dst_buffer(values_count);
    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> lock(cldnn_mem_src, *stream);
        std::memcpy(dst_buffer.data(), lock.data(), values_bytes_count);
    }

    bool are_equal = std::equal(src_buffer.begin(), src_buffer.begin() + values_count, dst_buffer.begin());
    ASSERT_TRUE(are_equal);
}

TEST(cl_mem_check, check_read_access_type) {
    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());
    cldnn::device::ptr device = devices.begin()->second;
    for (auto& dev : devices) {
        if (dev.second->get_info().dev_type == device_type::discrete_gpu)
            device = dev.second;
    }
    if (device->get_info().dev_type == device_type::integrated_gpu) {
        GTEST_SKIP();
    }

    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);
    auto stream = engine->create_stream({});

    size_t values_count = 100;
    size_t values_bytes_count = values_count * sizeof(float);
    std::vector<float> src_buffer(values_count);
    std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

    cldnn::layout linear_layout{{ov::Dimension(values_count)}, cldnn::data_types::f32, cldnn::format::bfyx};
    auto cldnn_mem_src = engine->allocate_memory(linear_layout, cldnn::allocation_type::cl_mem);
    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::write> lock(cldnn_mem_src, *stream);
        std::copy(src_buffer.begin(), src_buffer.end(), lock.data());
    }

    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> lock(cldnn_mem_src, *stream);
        std::copy(src_buffer.rbegin(), src_buffer.rend(), lock.data());
    }

    std::vector<float> dst_buffer(values_count);
    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> lock(cldnn_mem_src, *stream);
        std::memcpy(dst_buffer.data(), lock.data(), values_bytes_count);
    }

    bool are_equal = std::equal(src_buffer.begin(), src_buffer.begin() + values_count, dst_buffer.begin());
    ASSERT_TRUE(are_equal);
}
