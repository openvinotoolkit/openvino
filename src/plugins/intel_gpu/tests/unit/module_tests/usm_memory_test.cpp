// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_common.hpp"
#include "runtime/ocl/ocl_base_event.hpp"

#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "activation_inst.h"
#include "convolution_inst.h"
#include "crop_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "reshape_inst.h"
#include "pass_manager.h"
#include "program_wrapper.h"

#include <memory>
#include <tuple>

using namespace cldnn;
using namespace ::tests;

#if defined __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include <ocl/ocl_wrapper.hpp>
struct usm_test_params{
    allocation_type type;
};

static void init_device_and_engine(std::shared_ptr<ocl::ocl_device>& device,
                                   std::shared_ptr<ocl::ocl_engine>& engine,
                                   bool& supports_usm) {
    // Find device, which supports USMs.
    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    for (const auto& d : devices) {
        if (d.second->get_mem_caps().supports_usm()) {
            device = std::dynamic_pointer_cast<ocl::ocl_device>(d.second);
            supports_usm = true;
            break;
        }
    }

    engine = std::dynamic_pointer_cast<ocl::ocl_engine>(engine::create(engine_types::ocl, runtime_types::ocl, device));
}

class BaseUSMTest : public ::testing::TestWithParam<usm_test_params> {
protected:
    std::shared_ptr<ocl::ocl_device> _device = nullptr;
    std::shared_ptr<ocl::ocl_engine> _engine = nullptr;
    bool _supports_usm = false;
public:
    void SetUp() override {
        init_device_and_engine(_device, _engine, _supports_usm);

        if (!_device || !_engine || !_supports_usm) {
            GTEST_SKIP();
        }
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
//    usm_test_params{ allocation_type::usm_shared}, // Unsupported
    usm_test_params{ allocation_type::usm_device},
}));

class copy_and_read_buffer : public BaseUSMTest {};
TEST_P(copy_and_read_buffer, basic) {
    auto p = GetParam();
    if (!supports_usm()) {
        return;
    }
    try {
        ocl::ocl_stream stream(*_engine, {});

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
            auto host_buf = _engine->allocate_memory(linear_layout, allocation_type::usm_host);
            {
                cldnn::mem_lock<float> lock(host_buf, stream);
                std::copy(src_buffer.begin(), src_buffer.end(), lock.data());
            }
            cldnn_mem_src->copy_from(stream, *host_buf, true);
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
//        usm_test_params{ allocation_type::usm_shared }, // Unsupported
        usm_test_params{ allocation_type::usm_device },
}));

class fill_buffer : public BaseUSMTest {};
TEST_P(fill_buffer, DISABLED_basic) {
    auto p = GetParam();
    if (!supports_usm()) {
        return;
    }
    try {
        ocl::ocl_stream stream(*_engine, {});
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
//        usm_test_params{ allocation_type::usm_shared }, // Unsupported
        usm_test_params{ allocation_type::usm_device },
}));


class copy_between_gpu_buffer_and_gpu_usm : public BaseUSMTest {};
TEST_P(copy_between_gpu_buffer_and_gpu_usm, basic) {
    auto p = GetParam();
    if (!supports_usm()) {
        return;
    }
    try {
        ocl::ocl_stream stream(*_engine, {});

        size_t values_count = 100;
        size_t values_bytes_count = values_count * sizeof(float);
        std::vector<float> src_buffer(values_count);
        std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

        cldnn::layout linear_layout = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1, 1, int32_t(values_count), 1));
        auto usm_host_src = _engine->allocate_memory(linear_layout, allocation_type::usm_host);

        // Fill usm_host_src memory.
        cldnn::mem_lock<float> lock(usm_host_src, stream);
        std::copy(src_buffer.begin(), src_buffer.end(), lock.data());

        // Create dst memory
        auto mem_dst = _engine->allocate_memory(linear_layout, p.type);

        // Fill dst memory
        switch (p.type) {
        case allocation_type::usm_host:
        case allocation_type::usm_shared:
        case allocation_type::usm_device:
        {
            mem_dst->copy_from(stream, *usm_host_src, true);
            break;
        }
        case allocation_type::cl_mem: {
            mem_dst->copy_from(stream, *usm_host_src, true);
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
            cldnn::mem_lock<float> lock(usm_host_src, stream);
            std::memcpy(dst_buffer.data(), lock.data(), values_bytes_count);
            break;
        }
        case allocation_type::usm_device:
        case allocation_type::cl_mem: {
            auto host_buf = _engine->allocate_memory(linear_layout, allocation_type::usm_host);
            host_buf->copy_from(stream, *mem_dst);
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

INSTANTIATE_TEST_SUITE_P(cldnn_usm, copy_between_gpu_buffer_and_gpu_usm, ::testing::ValuesIn(std::vector<usm_test_params>{
    usm_test_params{ allocation_type::cl_mem },
    usm_test_params{ allocation_type::usm_host },
    usm_test_params{ allocation_type::usm_device },
}));

struct mem_test_params {
    size_t src_offset;
    size_t dst_offset;
    size_t size;
};

class offset_copy_host : public ::testing::TestWithParam<std::tuple<allocation_type, mem_test_params, bool>> {
protected:
    std::shared_ptr<ocl::ocl_device> _device = nullptr;
    std::shared_ptr<ocl::ocl_engine> _engine = nullptr;
    bool _supports_usm = false;
public:
    void SetUp() override {
        init_device_and_engine(_device, _engine, _supports_usm);

        if (!_device || !_engine || !_supports_usm) {
            GTEST_SKIP();
        }
    }

    bool supports_usm() const { return _supports_usm; }
};

class offset_copy : public ::testing::TestWithParam<std::tuple<allocation_type, allocation_type, mem_test_params, bool>> {
protected:
    std::shared_ptr<ocl::ocl_device> _device = nullptr;
    std::shared_ptr<ocl::ocl_engine> _engine = nullptr;
    bool _supports_usm = false;
public:
    void SetUp() override {
        init_device_and_engine(_device, _engine, _supports_usm);

        if (!_device || !_engine || !_supports_usm) {
            GTEST_SKIP();
        }
    }

    bool supports_usm() const { return _supports_usm; }
};

TEST_P(offset_copy, basic) {
    allocation_type src_allocation_type;
    allocation_type dst_allocation_type;
    mem_test_params params;
    bool use_copy_to;
    std::tie(src_allocation_type, dst_allocation_type, params, use_copy_to) = GetParam();

    const auto copy_size = params.size;
    const auto src_size = params.src_offset + copy_size;
    const auto dst_size = params.dst_offset + copy_size;

    auto stream = ocl::ocl_stream(*_engine, {});
    const auto src_layout = cldnn::layout({static_cast<int64_t>(src_size)}, cldnn::data_types::u8, cldnn::format::bfyx);
    const auto dst_layout = cldnn::layout({static_cast<int64_t>(dst_size)}, cldnn::data_types::u8, cldnn::format::bfyx);

    std::vector<uint8_t> src_buffer(src_size);
    for (size_t i = 0; i < src_size; i++)
        src_buffer[i] = i % 64;

    // Allocate and fill src memory
    auto src_memory = _engine->allocate_memory(src_layout, src_allocation_type);

    {
        const auto src_offset = 0;
        const auto dst_offset = 0;
        src_memory->copy_from(stream, src_buffer.data(), src_offset, dst_offset, src_size, true);
    }

    // Create dst memory and copy data
    auto dst_memory = _engine->allocate_memory(dst_layout, dst_allocation_type);

    if (use_copy_to) {
        src_memory->copy_to(stream, *dst_memory, params.src_offset, params.dst_offset, copy_size, true);
    } else {
        dst_memory->copy_from(stream, *src_memory, params.src_offset, params.dst_offset, copy_size, true);
    }

    // Read from dst mem
    std::vector<uint8_t> dst_buffer(copy_size);

    {
        const auto src_offset = params.dst_offset;
        const auto dst_offset = 0;
        dst_memory->copy_to(stream, dst_buffer.data(), src_offset, dst_offset, copy_size, true);
    }

    for (size_t i = 0; i < copy_size; i++) {
        ASSERT_EQ(src_buffer[i + params.src_offset], dst_buffer[i]) << i << "\n";
    }
}

TEST_P(offset_copy_host, basic) {
    allocation_type allocation_type;
    mem_test_params params;
    bool use_copy_to;
    std::tie(allocation_type, params, use_copy_to) = GetParam();

    const auto copy_size = params.size;
    const auto src_size = params.src_offset + copy_size;
    const auto mem_size = params.dst_offset + copy_size;

    auto stream = ocl::ocl_stream(*_engine, {});
    const auto mem_layout = cldnn::layout({static_cast<int64_t>(mem_size)}, cldnn::data_types::u8, cldnn::format::bfyx);

    std::vector<uint8_t> src_buffer(src_size);
    std::vector<uint8_t> dst_buffer(src_size);
    for (size_t i = 0; i < src_size; i++)
        src_buffer[i] = i % 64;

    auto memory = _engine->allocate_memory(mem_layout, allocation_type);
    memory->copy_from(stream, src_buffer.data(), params.src_offset, params.dst_offset, copy_size, true);
    memory->copy_to(stream, dst_buffer.data(), params.dst_offset, params.src_offset, copy_size, true);

    for (size_t i = 0; i < copy_size; i++) {
        ASSERT_EQ(src_buffer[i + params.src_offset], dst_buffer[i + params.src_offset]) << i << "\n";
    }
}

static std::vector<allocation_type> test_memory_types { allocation_type::cl_mem,
                                                        allocation_type::usm_host,
                                                        allocation_type::usm_device };

// clang-format off
INSTANTIATE_TEST_SUITE_P(mem_test,
                         offset_copy,
                         ::testing::Combine(::testing::ValuesIn(test_memory_types),
                                            ::testing::ValuesIn(test_memory_types),
                                            ::testing::Values(mem_test_params{0, 0, 381},
                                                              mem_test_params{100, 0, 381},
                                                              mem_test_params{0, 79, 381},
                                                              mem_test_params{100, 79, 381}),
                                            ::testing::Values(false, true)));

// clang-format off
INSTANTIATE_TEST_SUITE_P(mem_test,
                         offset_copy_host,
                         ::testing::Combine(::testing::ValuesIn(test_memory_types),
                                            ::testing::Values(mem_test_params{0, 0, 381},
                                                              mem_test_params{100, 0, 381},
                                                              mem_test_params{0, 79, 381},
                                                              mem_test_params{100, 79, 381}),
                                            ::testing::Values(false, true)));

TEST(mem_test, copy_small_buf_to_large_with_out_of_bound_access) {
    auto& ocl_engine = dynamic_cast<ocl::ocl_engine&>(get_test_engine());
    auto& stream = get_test_stream();
    auto small_buffer_size = 2048;
    auto large_buffer_size = 3072;

    auto small_buffer = ocl_engine.allocate_memory({{small_buffer_size}, data_types::u8, format::bfyx}, allocation_type::cl_mem, false);
    auto large_buffer = ocl_engine.allocate_memory({{large_buffer_size}, data_types::u8, format::bfyx}, allocation_type::cl_mem, false);

    OV_ASSERT_NO_THROW(small_buffer->copy_to(stream, *large_buffer, true));
}
