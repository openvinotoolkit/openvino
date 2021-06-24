// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/device_query.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_common.hpp"
#include "runtime/ocl/ocl_base_event.hpp"

#include "program_impl.h"
#include "topology_impl.h"
#include "data_inst.h"
#include "activation_inst.h"
#include "convolution_inst.h"
#include "crop_inst.h"
#include "network_impl.h"
#include "reshape_inst.h"
#include "pass_manager.h"
#include "program_impl_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

#if defined __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include <ocl/ocl_wrapper.hpp>

using namespace cldnn;
using namespace ::tests;

struct usm_test_params{
    allocation_type type;
};

class BaseUSMTest : public ::testing::TestWithParam<usm_test_params> {
protected:
    std::shared_ptr<ocl::ocl_device> _device = nullptr;
    std::shared_ptr<ocl::ocl_engine> _engine = nullptr;
    bool _supports_usm = false;
public:
    void SetUp() override {
        // Find device, which supports USMs.
        device_query query(engine_types::ocl, runtime_types::ocl);
        auto devices = query.get_available_devices();
        for (const auto& d : devices) {
            if (d.second->get_mem_caps().supports_usm()) {
                _device = std::dynamic_pointer_cast<ocl::ocl_device>(d.second);
                break;
            }
        }
        if (!_device) {
            GTEST_SUCCEED();
        }
        _engine = std::dynamic_pointer_cast<ocl::ocl_engine>(engine::create(engine_types::ocl, runtime_types::ocl, _device));
        _supports_usm = true;
    }

    bool supports_usm() const { return _supports_usm; }
};


class ctor_test : public BaseUSMTest {};
TEST_P(ctor_test, basic) {
    auto p = GetParam();
    if (!supports_usm()) {
        return;
    }
    try {
        cl::UsmMemory mem(_engine->get_usm_helper());
        switch (p.type) {
        case allocation_type::usm_host: {
            mem.allocateHost(1);
            break;
        }
        case allocation_type::usm_shared: {
            mem.allocateShared(1);
            break;
        }
        case allocation_type::usm_device: {
            mem.allocateDevice(1);
            break;
        }
        default:
            FAIL() << "Not supported allocation type!";
        }
        ASSERT_NE(nullptr, mem.get());
    }
    catch (...) {
        FAIL() << "Test failed, ctor of usm mems failed.";
    }
}

INSTANTIATE_TEST_SUITE_P(cldnn_usm, ctor_test, ::testing::ValuesIn(std::vector<usm_test_params>{
    usm_test_params{ allocation_type::usm_host},
    usm_test_params{ allocation_type::usm_shared},
    usm_test_params{ allocation_type::usm_device},
}));

class copy_and_read_buffer : public BaseUSMTest {};
TEST_P(copy_and_read_buffer, basic) {
    auto p = GetParam();
    if (!supports_usm()) {
        return;
    }
    try {
        ocl::ocl_stream stream(*_engine);

        size_t values_count = 100;
        size_t values_bytes_count = values_count * sizeof(float);
        std::vector<float> src_buffer(values_count);
        std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);
        cldnn::layout linear_layout = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1, 1, int32_t(values_count), 1));
        auto cldnn_mem_src = _engine->allocate_memory(linear_layout, p.type);

        // Fill src buffer
        switch (p.type) {
        case allocation_type::usm_host:
        case allocation_type::usm_shared: {
            cldnn::mem_lock<float> lock(cldnn_mem_src, stream);
            std::copy(src_buffer.begin(), src_buffer.end(), lock.data());
            break;
        }
        case allocation_type::usm_device: {
            auto casted = std::dynamic_pointer_cast<ocl::gpu_usm>(cldnn_mem_src);
            auto host_buf = _engine->allocate_memory(linear_layout, allocation_type::usm_host);
            {
                cldnn::mem_lock<float> lock(host_buf, stream);
                std::copy(src_buffer.begin(), src_buffer.end(), lock.data());
            }
            casted->copy_from(stream, *host_buf);
            break;
        }
        default:
            FAIL() << "Not supported allocation type!";
        }

        // Read from src buffer
        std::vector<float> dst_buffer(values_count);
        switch (p.type) {
        case allocation_type::usm_host:
        case allocation_type::usm_shared: {
            cldnn::mem_lock<float> lock(cldnn_mem_src, stream);
            std::memcpy(dst_buffer.data(), lock.data(), values_bytes_count);
            break;
        }
        case allocation_type::usm_device: {
            auto host_buf = _engine->allocate_memory(linear_layout, allocation_type::usm_host);
            host_buf->copy_from(stream, *cldnn_mem_src);
            {
                cldnn::mem_lock<float> lock(host_buf, stream);
                std::memcpy(dst_buffer.data(), lock.data(), values_bytes_count);
            }
            break;
        }
        default:
            FAIL() << "Not supported allocation type!";
        }
        bool are_equal = std::equal(src_buffer.begin(), src_buffer.begin() + 100, dst_buffer.begin());
        ASSERT_EQ(true, are_equal);
    } catch (const char* msg) {
        FAIL() << msg;
    }

}

INSTANTIATE_TEST_SUITE_P(cldnn_usm, copy_and_read_buffer, ::testing::ValuesIn(std::vector<usm_test_params>{
        usm_test_params{ allocation_type::usm_host },
        usm_test_params{ allocation_type::usm_shared },
        usm_test_params{ allocation_type::usm_device },
}));

class fill_buffer : public BaseUSMTest {};
TEST_P(fill_buffer, DISABLED_basic) {
    auto p = GetParam();
    if (!supports_usm()) {
        return;
    }
    try {
        ocl::ocl_stream stream(*_engine);
        auto queue = stream.get_cl_queue();
        auto usm_helper = stream.get_usm_helper();

        size_t values_count = 100;
        size_t values_bytes_count = values_count * sizeof(float);
        cl::UsmMemory mem(usm_helper);
        switch (p.type) {
        case allocation_type::usm_host:
            mem.allocateHost(values_bytes_count);
            break;
        case allocation_type::usm_shared:
            mem.allocateShared(values_bytes_count);
            break;
        case allocation_type::usm_device:
            mem.allocateDevice(values_bytes_count);
            break;
        default:
            FAIL() << "Not supported allocation type!";
        }
        // Fill buffer !! This can fail with old driver, which does not support fill usm api.
        cl::Event ev;
        unsigned char pattern = 0;
        usm_helper.enqueue_fill_mem(
            queue,
            mem.get(),
            static_cast<const void*>(&pattern),
            sizeof(unsigned char),
            values_bytes_count,
            nullptr,
            &ev
        );
        ev.wait();

        // Read from src buffer
        std::vector<float> dst_buffer(values_count);
        std::iota(dst_buffer.begin(), dst_buffer.end(), 5.0f); //fill with other value, so we can easily compare with 0.0f
        auto values_ptr = mem.get();
        switch (p.type) {
        case allocation_type::usm_host:
        case allocation_type::usm_shared: {
            std::memcpy(dst_buffer.data(), values_ptr, values_bytes_count);
            break;
        }
        case allocation_type::usm_device: {
            cl::UsmMemory host_mem(usm_helper);
            host_mem.allocateHost(values_bytes_count);
            usm_helper.enqueue_memcpy(
                queue,
                host_mem.get(),
                mem.get(),
                values_bytes_count,
                true
            );
            auto host_ptr = host_mem.get();
            std::memcpy(dst_buffer.data(), host_ptr, values_bytes_count);
            break;
        }
        default:
            FAIL() << "Not supported allocation type!";
        }
        bool are_equal = std::all_of(dst_buffer.begin(), dst_buffer.begin() + values_count, [](float f) {return f == 0; });
        ASSERT_EQ(true, are_equal);
    }
    catch (const char* msg) {
        FAIL() << msg;
    }

}

INSTANTIATE_TEST_SUITE_P(cldnn_usm, fill_buffer, ::testing::ValuesIn(std::vector<usm_test_params>{
    usm_test_params{ allocation_type::usm_host },
        usm_test_params{ allocation_type::usm_shared },
        usm_test_params{ allocation_type::usm_device },
}));
