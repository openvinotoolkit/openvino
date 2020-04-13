/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <memory>

#include <gtest/gtest.h>

#include "program_impl.h"
#include "topology_impl.h"
#include "engine_impl.h"
#include "memory_impl.h"
#include "data_inst.h"
#include "activation_inst.h"
#include "convolution_inst.h"
#include "crop_inst.h"
#include "network_impl.h"
#include "reshape_inst.h"
#include "pass_manager.h"
#include "api/engine.hpp"
#include "test_utils.h"
#include "program_impl_wrapper.h"
#include "gpu/ocl_queue_wrapper.h"
#include "gpu/memory_gpu.h"
#include "gpu/ocl_toolkit.h"
#include "gpu/command_queues_builder.h"
#include "gpu/ocl_base_event.h"

using namespace cldnn;
using namespace ::tests;

#if defined __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include <cl2_wrapper.h>

using namespace cldnn;
using namespace ::tests;

struct usm_test_params{
    allocation_type type;
};

class BaseUSMTest : public ::testing::TestWithParam<usm_test_params> {
protected:
    std::shared_ptr<device> _device = nullptr;
    std::shared_ptr<engine> _engine = nullptr;
    bool _supports_usm = false;
public:
    void SetUp() override {
        // Find device, which supports USMs.
        device_query query;
        auto devices = query.get_available_devices();
        for (const auto& d : devices) {
            if (d.second.get()->mem_caps().supports_usm()) {
                _device = std::make_shared<device>(d.second);
                break;
            }
        }
        _engine = std::make_shared<engine>(_device->get());
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
        cl::UsmMemory mem(_device->get()->get_context());
        auto cl_dev = _device->get()->get_device();
        switch (p.type) {
        case allocation_type::usm_host: {
            mem.allocateHost(1);
            break;
        }
        case allocation_type::usm_shared: {
            mem.allocateShared(cl_dev, 1);
            break;
        }
        case allocation_type::usm_device: {
            mem.allocateDevice(cl_dev, 1);
            break;
        }
        default:
            FAIL() << "Not supported allocation type!";
        }
        ASSERT_NE(nullptr, mem.get());
        ASSERT_EQ(mem.use_count(), 1);
    }
    catch (...) {
        FAIL() << "Test failed, ctor of usm mems failed.";
    }
}

INSTANTIATE_TEST_CASE_P(cldnn_usm, ctor_test, ::testing::ValuesIn(std::vector<usm_test_params>{
    usm_test_params{ allocation_type::usm_host},
    usm_test_params{ allocation_type::usm_shared},
    usm_test_params{ allocation_type::usm_device},
}), );

class copy_and_read_buffer : public BaseUSMTest {};
TEST_P(copy_and_read_buffer, basic) {
    auto p = GetParam();
    if (!supports_usm()) {
        return;
    }
    try {
        gpu::command_queues_builder q_builder(_device->get()->get_context(), _device->get()->get_device(), _device->get()->get_platform());
        q_builder.build();
        auto queue = cl::CommandQueueIntel(q_builder.queue());

        size_t values_count = 100;
        size_t values_bytes_count = values_count * sizeof(float);
        std::vector<float> src_buffer(values_count);
        std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);
        cldnn::layout linear_layout = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1, 1, int32_t(values_count), 1));
        auto cldnn_mem_src = _engine->get()->allocate_memory(linear_layout, p.type);
        auto ptr_to_fill = cldnn_mem_src->lock();
        // Fill src buffer
        switch (p.type) {
        case allocation_type::usm_host:
        case allocation_type::usm_shared: {
            std::copy(src_buffer.begin(), src_buffer.end(), static_cast<float*>(ptr_to_fill));
            cldnn_mem_src->unlock();
            break;
        }
        case allocation_type::usm_device: {
            auto host_buf = _engine->get()->allocate_memory(linear_layout, allocation_type::usm_host);
            std::copy(src_buffer.begin(), src_buffer.end(), static_cast<float*>(host_buf->lock()));
            host_buf->unlock();
            queue.enqueueCopyUsm(
                dynamic_cast<gpu::gpu_usm&>(*host_buf).get_buffer(),
                dynamic_cast<gpu::gpu_usm&>(*cldnn_mem_src).get_buffer(),
                values_bytes_count,
                true
            );
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
            auto values_ptr = cldnn_mem_src->lock();
            std::memcpy(dst_buffer.data(), values_ptr, values_bytes_count);
            cldnn_mem_src->unlock();
            break;
        }
        case allocation_type::usm_device: {
            auto host_buf = _engine->get()->allocate_memory(linear_layout, allocation_type::usm_host);
            queue.enqueueCopyUsm(
                dynamic_cast<gpu::gpu_usm&>(*cldnn_mem_src).get_buffer(),
                dynamic_cast<gpu::gpu_usm&>(*host_buf).get_buffer(),
                values_bytes_count,
                true
                );
            auto values_ptr = host_buf->lock();
            std::memcpy(dst_buffer.data(), values_ptr, values_bytes_count);
            host_buf->unlock();
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

INSTANTIATE_TEST_CASE_P(cldnn_usm, copy_and_read_buffer, ::testing::ValuesIn(std::vector<usm_test_params>{
        usm_test_params{ allocation_type::usm_host },
        usm_test_params{ allocation_type::usm_shared },
        usm_test_params{ allocation_type::usm_device },
}), );

class fill_buffer : public BaseUSMTest {};
TEST_P(fill_buffer, DISABLED_basic) {
    auto p = GetParam();
    if (!supports_usm()) {
        return;
    }
    try {
        gpu::command_queues_builder q_builder(_device->get()->get_context(), _device->get()->get_device(), _device->get()->get_platform());
        q_builder.build();
        auto queue = cl::CommandQueueIntel(q_builder.queue());

        size_t values_count = 100;
        size_t values_bytes_count = values_count * sizeof(float);
        cl::UsmMemory mem(_device->get()->get_context());
        switch (p.type) {
        case allocation_type::usm_host:
            mem.allocateHost(values_bytes_count);
            break;
        case allocation_type::usm_shared:
            mem.allocateShared(_device->get()->get_device(), values_bytes_count);
            break;
        case allocation_type::usm_device:
            mem.allocateDevice(_device->get()->get_device(), values_bytes_count);
            break;
        default:
            FAIL() << "Not supported allocation type!";
        }
        // Fill buffer !! This can fail with old driver, which does not support fill usm api.
        cl::Event ev;
        unsigned char pattern = 0;
        queue.enqueueFillUsm<unsigned char>(
            mem,
            pattern,
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
            cl::UsmMemory host_mem(_device->get()->get_context());
            host_mem.allocateHost(values_bytes_count);
            queue.enqueueCopyUsm(
                mem,
                host_mem,
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

INSTANTIATE_TEST_CASE_P(cldnn_usm, fill_buffer, ::testing::ValuesIn(std::vector<usm_test_params>{
    usm_test_params{ allocation_type::usm_host },
        usm_test_params{ allocation_type::usm_shared },
        usm_test_params{ allocation_type::usm_device },
}), );
