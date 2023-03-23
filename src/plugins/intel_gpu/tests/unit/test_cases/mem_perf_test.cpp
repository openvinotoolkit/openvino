// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "opencl_helper_instance.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>

static std::string mem_latency_kernel_code =
    "\n#ifndef INTEL_INTERNAL_DEBUG_H_INCLUDED "
    "\n#define INTEL_INTERNAL_DEBUG_H_INCLUDED "
    "\nulong __attribute__((overloadable)) intel_get_cycle_counter( void ); "
    "\n#endif "
    ""
    "\n__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))"
    "\nvoid kernel touch_data(__global uint* dst, const __global int* test_data) {"
    "\n    int sum = 0;"
    "\n    int val = 0;"
    "\n    int next_id = 0;"
    "\n    const uint gid = get_global_id(0);"
    "\n    __attribute__((opencl_unroll_hint(1)))"
    "\n    for (uint j = 0; j < LOAD_ITERATIONS; j++) {"
    "\n        // val = intel_sub_group_block_read(test_data + (next_id));"
    "\n        val = test_data[next_id + gid];"
    "\n        sum += val;"
    "\n        next_id = val;"
    "\n    }"
    "\n    dst[gid] = sum;"
    "\n}"
    ""
    "\n__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))"
    "\nvoid kernel latency_test(__global uint* dst, const __global int* test_data, int iteration_num) {"
    "\n    int sum = 0;"
    "\n    int val = 0;"
    "\n    int next_id = 0;"
    "\n    ulong timer_start = 0;"
    "\n    ulong timer_end = 0, total_time = 0, time = 0;"
    "\n    ulong max_time = 0, min_time = -1;"
    "\n    const uint gid = get_global_id(0);"
    "\n    __attribute__((opencl_unroll_hint(1)))"
    "\n#ifdef USE_PRE_HEAT"
    "\n    for (uint iter = 0; iter < iteration_num + 1; iter++) {"
    "\n#else"
    "\n    for (uint iter = 0; iter < iteration_num; iter++) {"
    "\n#endif"
    "\n        next_id = 0;"
    "\n        __attribute__((opencl_unroll_hint(1)))"
    "\n        for (uint j = 0; j < LOAD_ITERATIONS; j++) {"
    "\n            timer_start = intel_get_cycle_counter();"
    "\n            val = test_data[next_id + gid];"
    "\n            // val = intel_sub_group_block_read(test_data + (next_id));"
    "\n            sum += val;"
    "\n            timer_end = intel_get_cycle_counter();"
    "\n            time = timer_end - timer_start;"
    "\n            max_time = max(max_time, time);"
    "\n            min_time = min(min_time, time);"
    "\n            total_time += time;"
    "\n            next_id = val;"
    "\n#ifdef USE_PRE_HEAT"
    "\n            if (iter == 0) {"
    "\n                total_time = 0;"
    "\n                max_time = 0;"
    "\n                min_time = -1;"
    "\n            }"
    "\n#endif"
    "\n        }"
    "\n    }"
    "\n    dst[gid] = sum;"
    "\n    dst[0] = total_time;"
    "\n    dst[1] = max_time;"
    "\n    dst[2] = min_time;"
    "\n}";

static size_t img_size = 800;
static std::string kernel_code =
    "__attribute__((intel_reqd_sub_group_size(16)))"
    "__attribute__((reqd_work_group_size(16, 1, 1)))"
    "void kernel simple_reorder(const __global uchar* src, __global float* dst) {"
    "    uint gid = get_global_id(0);"
    "    dst[gid] = convert_float(src[gid]) * 0.33f;"
    "}";
static size_t max_iter = 1000;

using time_interval = std::chrono::microseconds;
static std::string time_suffix = "us";

static void printTimings(double avg, int64_t max) {
    std::cout << "img_size=" << img_size << " iters=" << max_iter << " exec time: avg="
              << avg << time_suffix << ", max=" << max << time_suffix << std::endl;
}

static void fill_input(uint8_t* ptr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        ptr[i] = static_cast<uint8_t>(i % 255);
    }
}

static void run_test(std::function<void()> preprocessing,
                     std::function<void()> body,
                     std::function<void()> postprocessing = [](){}) {
    using Time = std::chrono::high_resolution_clock;
    int64_t max_time = 0;
    double avg_time = 0.0;
    for (size_t iter = 0; iter < max_iter; iter++) {
        preprocessing();
        auto start = Time::now();
        body();
        auto stop = Time::now();
        std::chrono::duration<float> fs = stop - start;
        time_interval d = std::chrono::duration_cast<time_interval>(fs);
        max_time = std::max(max_time, static_cast<int64_t>(d.count()));
        avg_time += static_cast<double>(d.count());
        postprocessing();
    }

    avg_time /= max_iter;

    printTimings(avg_time, max_time);
}

static void validate_result(float* res_ptr, size_t size) {
    for (size_t i = 0; i < size; i++) {
      ASSERT_EQ(res_ptr[i], static_cast<float>(i % 255) * 0.33f) << "i=" << i;
    }

    std::cout << "accuracy: OK\n";
}

TEST(mem_perf_test_to_device, DISABLED_fill_input) {
    auto ocl_instance = std::make_shared<OpenCL>();
    cl::UsmMemory input_buffer(*ocl_instance->_usm_helper);
    input_buffer.allocateHost(sizeof(uint8_t) * img_size * img_size);

    std::cout << "Time of host buffer filling" << std::endl;

    run_test([](){}, [&]() {
        fill_input(static_cast<uint8_t*>(input_buffer.get()), img_size * img_size);
    });
}

TEST(mem_perf_test_to_device, DISABLED_buffer_no_lock) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    std::cout << "Time of kernel execution" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::CommandQueue queue(ctx, device);

    run_test([](){}, [&]() {
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });
}

TEST(mem_perf_test_to_device, DISABLED_buffer_lock_rw) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    std::cout << "Time of copying data from mapped to host cl::Buffer (ReadWrite access modifier) to device memory" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::CommandQueue queue(ctx, device);

    void* _mapped_ptr = nullptr;
    run_test([&](){
        _mapped_ptr = queue.enqueueMapBuffer(input_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(uint8_t) * img_size * img_size, nullptr, nullptr);
        fill_input(static_cast<uint8_t*>(_mapped_ptr), img_size * img_size);
    }, [&]() {
        queue.enqueueUnmapMemObject(input_buffer, _mapped_ptr);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });

    _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    validate_result(static_cast<float*>(_mapped_ptr), img_size * img_size);
    queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
}

TEST(mem_perf_test_to_device, DISABLED_buffer_lock_w) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    std::cout << "Time of copying data from mapped to host cl::Buffer (Write access modifier) to device memory" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::CommandQueue queue(ctx, device);

    void* _mapped_ptr = nullptr;
    run_test([&](){
        _mapped_ptr = queue.enqueueMapBuffer(input_buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(uint8_t) * img_size * img_size, nullptr, nullptr);
        fill_input(static_cast<uint8_t*>(_mapped_ptr), img_size * img_size);
    }, [&]() {
        queue.enqueueUnmapMemObject(input_buffer, _mapped_ptr);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });

    _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    validate_result(static_cast<float*>(_mapped_ptr), img_size * img_size);
    queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
}

TEST(mem_perf_test_to_device, DISABLED_buffer_copy) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    std::cout << "Time of copying data from host buffer (std::vector) to cl::Buffer located in device memory" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::CommandQueue queue(ctx, device);
    std::vector<uint8_t> input(img_size*img_size);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input.data()), img_size * img_size);
    }, [&]() {
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, false, 0, img_size*img_size, input.data(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    });

    auto _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    validate_result(static_cast<float*>(_mapped_ptr), img_size * img_size);
    queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
}

TEST(mem_perf_test_to_device, DISABLED_buffer_copy_usm_host) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from host buffer cl::UsmMemory (UsmHost type) to cl::Buffer located in device memory" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::UsmMemory input_buffer_host(*ocl_instance->_usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
    }, [&]() {
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, false, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    });

    auto _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    validate_result(static_cast<float*>(_mapped_ptr), img_size * img_size);
    queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
}

TEST(mem_perf_test_to_device, DISABLED_usm_host) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;
    auto& usm_helper = *ocl_instance->_usm_helper;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of transfering data from host buffer cl::UsmMemory (UsmHost type) to device" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::UsmMemory input_buffer(usm_helper);
    input_buffer.allocateHost(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory output_buffer(usm_helper);
    output_buffer.allocateDevice(sizeof(float) * img_size * img_size);
    cl::UsmMemory output_buffer_host(usm_helper);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);
    cl::Kernel kernel1(program, "simple_reorder");
    cl::KernelIntel kernel(kernel1, usm_helper);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer.get()), img_size * img_size);
    }, [&]() {
        kernel.setArgUsm(0, input_buffer);
        kernel.setArgUsm(1, output_buffer);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });

    usm_helper.enqueue_memcpy(queue,
                              output_buffer_host.get(),
                              output_buffer.get(),
                              sizeof(float) * img_size * img_size,
                              true,
                              nullptr,
                              nullptr);
    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}

TEST(mem_perf_test_to_device, DISABLED_usm_device) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;
    auto& usm_helper = *ocl_instance->_usm_helper;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from device buffer cl::UsmMemory (UsmDevice type) to cl::UsmMemory (UsmDevice type)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::UsmMemory input_buffer_host(usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory input_buffer_device(usm_helper);
    input_buffer_device.allocateDevice(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory input_buffer_device_second(usm_helper);
    input_buffer_device_second.allocateDevice(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory output_buffer(usm_helper);
    output_buffer.allocateDevice(sizeof(float) * img_size * img_size);
    cl::UsmMemory output_buffer_host(usm_helper);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);
    cl::Kernel kernel1(program, "simple_reorder");
    cl::KernelIntel kernel(kernel1, usm_helper);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        usm_helper.enqueue_memcpy(queue,
                                  input_buffer_device.get(),
                                  input_buffer_host.get(),
                                  img_size * img_size,
                                  true,
                                  nullptr,
                                  nullptr);
    }, [&]() {
        cl::Event copy_ev;
        usm_helper.enqueue_memcpy(queue,
                                  input_buffer_device_second.get(),
                                  input_buffer_device.get(),
                                  img_size * img_size,
                                  false,
                                  nullptr,
                                  &copy_ev);

        kernel.setArgUsm(0, input_buffer_device_second);
        kernel.setArgUsm(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    });

    usm_helper.enqueue_memcpy(queue,
                              output_buffer_host.get(),
                              output_buffer.get(),
                              sizeof(float) * img_size * img_size,
                              true,
                              nullptr,
                              nullptr);
    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}

TEST(mem_perf_test_to_device, DISABLED_usm_device_copy) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;
    auto& usm_helper = *ocl_instance->_usm_helper;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from host buffer cl::UsmMemory (UsmHost type) to cl::UsmMemory (UsmDevice type)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::UsmMemory input_buffer_host(usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory input_buffer_device(usm_helper);
    input_buffer_device.allocateDevice(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory output_buffer(usm_helper);
    output_buffer.allocateDevice(sizeof(float) * img_size * img_size);
    cl::UsmMemory output_buffer_host(usm_helper);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);
    cl::Kernel kernel1(program, "simple_reorder");
    cl::KernelIntel kernel(kernel1, usm_helper);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
    }, [&]() {
        cl::Event copy_ev;
        usm_helper.enqueue_memcpy(queue,
                                  input_buffer_device.get(),
                                  input_buffer_host.get(),
                                  sizeof(uint8_t) * img_size * img_size,
                                  false,
                                  nullptr,
                                  &copy_ev);
        kernel.setArgUsm(0, input_buffer_device);
        kernel.setArgUsm(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    });

    usm_helper.enqueue_memcpy(queue,
                              output_buffer_host.get(),
                              output_buffer.get(),
                              sizeof(float) * img_size * img_size,
                              true,
                              nullptr,
                              nullptr);
    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}

TEST(mem_perf_test_to_device, DISABLED_cl_buffer_to_usm_device) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;
    auto& usm_helper = *ocl_instance->_usm_helper;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of kernel execution w/o copying the data (input buffer is cl::Buffer located in device memory)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory input_buffer_host(usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory output_buffer_device(usm_helper);
    output_buffer_device.allocateDevice(sizeof(float) * img_size * img_size);
    cl::UsmMemory output_buffer_host(usm_helper);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);

    cl::Kernel kernel1(program, "simple_reorder");
    cl::KernelIntel kernel(kernel1, usm_helper);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, img_size*img_size, input_buffer_host.get(), nullptr, nullptr);
    }, [&]() {
        kernel.setArg(0, input_buffer);
        kernel.setArgUsm(1, output_buffer_device);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });

    usm_helper.enqueue_memcpy(queue,
                              output_buffer_host.get(),
                              output_buffer_device.get(),
                              sizeof(float) * img_size * img_size,
                              true,
                              nullptr,
                              nullptr);
    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}

TEST(mem_perf_test_to_host, DISABLED_buffer_lock_rw) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from device buffer (cl::Buffer) to host via buffer mapping (ReadWrite access modifier)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::UsmMemory input_buffer_host(*ocl_instance->_usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);

    cl::CommandQueue queue(ctx, device);

    void* _mapped_ptr = nullptr;

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    }, [&]() {
        _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    }, [&]() {
        queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
    });

    _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_WRITE | CL_MAP_WRITE, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    validate_result(static_cast<float*>(_mapped_ptr), img_size * img_size);
    queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
}

TEST(mem_perf_test_to_host, DISABLED_buffer_lock_r) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from device buffer (cl::Buffer) to host via buffer mapping (Read access modifier)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::UsmMemory input_buffer_host(*ocl_instance->_usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);

    cl::CommandQueue queue(ctx, device);

    fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
    cl::Event copy_ev;
    queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);
    cl::Event ev;
    std::vector<cl::Event> dep_ev = {copy_ev};
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
    cl::WaitForEvents({ev});

    void* _mapped_ptr = nullptr;

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    }, [&]() {
        _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    }, [&](){
        queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
    });

    _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    validate_result(static_cast<float*>(_mapped_ptr), img_size * img_size);
    queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
}

TEST(mem_perf_test_to_host, DISABLED_buffer_copy_usm_host_ptr_blocking_r) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from device buffer (cl::Buffer) to host buffer cl::UsmMemory (UsmHost type) - Bloking call" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::UsmMemory input_buffer_host(*ocl_instance->_usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);

    cl::UsmMemory output_buffer_host(*ocl_instance->_usm_helper);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    }, [&]() {
        queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float)*img_size*img_size, output_buffer_host.get());
    });

    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}

TEST(mem_perf_test_to_host, DISABLED_buffer_copy_usm_host_ptr_events_r) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from device buffer (cl::Buffer) to host buffer cl::UsmMemory (UsmHost type) - Non-blocling call (events)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::UsmMemory input_buffer_host(*ocl_instance->_usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);

    cl::UsmMemory output_buffer_host(*ocl_instance->_usm_helper);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    }, [&]() {
        cl::Event copy_ev;
        queue.enqueueReadBuffer(output_buffer, CL_FALSE, 0, sizeof(float)*img_size*img_size, output_buffer_host.get(), nullptr, &copy_ev);
        cl::WaitForEvents({copy_ev});
    });

    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}

TEST(mem_perf_test_to_host, DISABLED_buffer_copy_host_ptr_events_r) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from device buffer (cl::Buffer) to host buffer (std::vector) - Non-blocling call (events)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::UsmMemory input_buffer_host(*ocl_instance->_usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);

    std::vector<float> output_buffer_host(img_size * img_size);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    }, [&]() {
        cl::Event copy_ev;
        queue.enqueueReadBuffer(output_buffer, CL_FALSE, 0, sizeof(float)*img_size*img_size, output_buffer_host.data(), nullptr, &copy_ev);
        cl::WaitForEvents({copy_ev});
    });

    validate_result(static_cast<float*>(output_buffer_host.data()), img_size * img_size);
}

TEST(mem_perf_test_to_host_and_back_to_device, DISABLED_buffer_copy_usm_host_ptr_events_rw) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from device buffer (cl::Buffer) to host buffer cl::UsmMemory (UsmHost type) "
              << "and back to device (cl::Buffer) - Non-blocling calls (events)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::UsmMemory input_buffer_host(*ocl_instance->_usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);

    cl::UsmMemory output_buffer_host(*ocl_instance->_usm_helper);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    }, [&]() {
        cl::Event to_host_ev, to_device_ev;
        queue.enqueueReadBuffer(output_buffer, CL_FALSE, 0, sizeof(float)*img_size*img_size, output_buffer_host.get(), nullptr, &to_host_ev);
        std::vector<cl::Event> copy_ev {to_host_ev};
        queue.enqueueWriteBuffer(output_buffer, CL_FALSE, 0, img_size*img_size, output_buffer_host.get(), &copy_ev, &to_device_ev);
        cl::WaitForEvents({to_device_ev});
    });

    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}

TEST(mem_perf_test_to_host_and_back_to_device, DISABLED_buffer_copy_host_ptr_events_rw) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        GTEST_SKIP();

    std::cout << "Time of copying data from device buffer (cl::Buffer) to host buffer (std::vector) and back to device (cl::Buffer) - Non-blocling calls (events)" << std::endl;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::UsmMemory input_buffer_host(*ocl_instance->_usm_helper);
    input_buffer_host.allocateHost(sizeof(uint8_t) * img_size * img_size);

    std::vector<float> output_buffer_host(img_size * img_size);

    cl::CommandQueue queue(ctx, device);

    run_test([&](){
        fill_input(static_cast<uint8_t*>(input_buffer_host.get()), img_size * img_size);
        cl::Event copy_ev;
        queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, img_size*img_size, input_buffer_host.get(), nullptr, &copy_ev);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        std::vector<cl::Event> dep_ev = {copy_ev};
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), &dep_ev, &ev);
        cl::WaitForEvents({ev});
    }, [&]() {
        cl::Event read_ev, write_ev;
        queue.enqueueReadBuffer(output_buffer, CL_FALSE, 0, sizeof(float)*img_size*img_size, output_buffer_host.data(), nullptr, &read_ev);
        std::vector<cl::Event> ev_list{read_ev};
        queue.enqueueWriteBuffer(output_buffer, CL_FALSE, 0, sizeof(float)*img_size*img_size, output_buffer_host.data(), &ev_list, &write_ev);
        cl::WaitForEvents({write_ev});
    });

    validate_result(static_cast<float*>(output_buffer_host.data()), img_size * img_size);
}

using MemTestParams = std::tuple<
    std::pair<size_t, size_t>,  // Buffer size in bytes, number of iterations in kernel
    size_t                      // Mode (0 - cold run, 1 - warm data in prev kernel, 2 - warm data in main kernel)
>;

struct PrintToStringParamName {
    std::string operator()(const testing::TestParamInfo<MemTestParams>& params) {
        size_t mode;
        std::pair<size_t, size_t> size_iters;
        std::tie(size_iters, mode) = params.param;
        size_t buffer_size_b = size_iters.first;
        size_t iters_num = size_iters.second;
        std::stringstream buf;
        buf << "buffer_size_" << buffer_size_b << "B"
            << "_mode_" << mode
            << "_iters_num_" << iters_num;
        return buf.str();
    }
};

struct latency_test
    : public ::testing::TestWithParam<MemTestParams> {
public:
    void test() {
        size_t mode;
        std::pair<size_t, size_t> size_iters;
        std::tie(size_iters, mode) = this->GetParam();
        size_t buffer_size_b = size_iters.first;
        size_t iters_num = size_iters.second;

        auto ocl_instance = std::make_shared<OpenCL>();
        auto& ctx = ocl_instance->_context;
        auto& device = ocl_instance->_device;

        if (!ocl_instance->_supports_usm)
            GTEST_SKIP();

        std::cout << "Run with size " << buffer_size_b / 1024.0 << "KB mode=" << mode << " iters=" << iters_num << "\n";

        const float gpu_frequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() / 1000.0;

        const bool use_two_kernels = mode == 1;
        const bool use_pre_heat_in_kernel = mode == 2;

        const size_t gws = 16;
        const size_t lws = 16;
        const size_t simd = 16;
        const size_t test_data_size = buffer_size_b;
        const size_t profiling_data_size = sizeof(uint32_t) * simd;
        const size_t load_instuction_count = test_data_size / simd / sizeof(uint32_t);
        const std::string touch_data_kernel = "touch_data";
        const std::string kernel_name = "latency_test";
        std::string build_options = "-DLOAD_ITERATIONS=" + std::to_string(load_instuction_count) + " "
                                  + "-DSUB_GROUP_SIZE=" + std::to_string(simd) + " ";
        if (use_pre_heat_in_kernel)
            build_options += "-DUSE_PRE_HEAT=1 ";

        std::string used_kernels = kernel_name;
        if (use_two_kernels)
            used_kernels = touch_data_kernel + " " + used_kernels;

        std::cout << "gpu_frequency=" << gpu_frequency << "GHz" << std::endl;
        std::cout << "gws=" << gws << " lws=" << lws << " simd=" << simd << " kernels=[" << used_kernels << "] use_pre_heat_in_kernel=" << use_pre_heat_in_kernel << std::endl;
        std::cout << "test_data_size=" << test_data_size / 1024.0 << "KB profiling_data_size=" << profiling_data_size << "B "
                  << "total_load_instructions_number=" << load_instuction_count << std::endl;

        cl::Program program(ctx, mem_latency_kernel_code);
        program.build({device}, build_options.c_str());
        std::stringstream dump_file;

        cl::Kernel latency_kernel_cl(program, kernel_name.c_str());
        cl::KernelIntel latency_kernel(latency_kernel_cl, *ocl_instance->_usm_helper);

        std::vector<int> test_data_buffer;
        for (size_t i = 0; i < test_data_size / sizeof(int); i++) {
            test_data_buffer.push_back(static_cast<int>(((i / simd) + 1) * simd));
        }

        cl::UsmMemory test_data_buffer_device(*ocl_instance->_usm_helper);
        test_data_buffer_device.allocateDevice(test_data_size);

        cl::UsmMemory output_buffer_device(*ocl_instance->_usm_helper);
        output_buffer_device.allocateDevice(profiling_data_size);

        cl::UsmMemory output_buffer_host(*ocl_instance->_usm_helper);
        output_buffer_host.allocateHost(profiling_data_size);

        cl::CommandQueue queue(ctx, device);

        ocl_instance->_usm_helper->enqueue_memcpy(queue, test_data_buffer_device.get(), test_data_buffer.data(), test_data_size, true);

        latency_kernel.setArgUsm(0, output_buffer_device);
        latency_kernel.setArgUsm(1, test_data_buffer_device);
        latency_kernel.setArg(2, iters_num);

        cl::Event ev1;
        std::vector<cl::Event> wait_list1;

        // Flush all caches
        queue.finish();

        if (use_two_kernels) {
            cl::Kernel cl_kernel1(program, touch_data_kernel.c_str());
            cl::KernelIntel kernel1(cl_kernel1, *ocl_instance->_usm_helper);

            kernel1.setArgUsm(0, output_buffer_device);
            kernel1.setArgUsm(1, test_data_buffer_device);

            queue.enqueueNDRangeKernel(kernel1, cl::NDRange(), cl::NDRange(gws), cl::NDRange(lws), nullptr, &ev1);
            wait_list1.push_back(ev1);
        }

        cl::Event ev2;
        queue.enqueueNDRangeKernel(latency_kernel, cl::NDRange(), cl::NDRange(gws), cl::NDRange(lws), &wait_list1, &ev2);
        cl::WaitForEvents({ev2});

        ocl_instance->_usm_helper->enqueue_memcpy(queue, output_buffer_host.get(), output_buffer_device.get(), test_data_size, true);

        uint32_t* profiling_res = static_cast<uint32_t*>(output_buffer_host.get());
        auto const clcs = profiling_res[0];
        auto const max = profiling_res[1];
        auto const max_ns = profiling_res[1] / gpu_frequency;
        auto const min = profiling_res[2];
        auto const min_ns = profiling_res[2] / gpu_frequency;
        std::cout << "max=" << max << "(" << max_ns << "ns)" << " min=" << min << "(" << min_ns << "ns)" << std::endl;
        std::cout << "clcs=" << clcs << " clcs_per_load=" << clcs / load_instuction_count / iters_num << " latency_per_load=" << clcs / load_instuction_count / gpu_frequency / iters_num << "ns" << std::endl;
    }
};

TEST_P(latency_test, DISABLED_benchmark) {
    ASSERT_NO_FATAL_FAILURE(test());
}

const size_t _B = 1;
const size_t _KB = 1024 * _B;
const size_t _MB = 1024 * _KB;

// Buffer size in bytes and number of iterations in kernel (for larger buffers the number of iterations is
// getting lower to prevent timer's variable overflow)
const std::vector<std::pair<size_t, size_t>> sizes_and_iters = {
    { 64 * _B, 10 },
    { 256 * _B, 10 },
    { 1 * _KB, 10 },
    { 2 * _KB, 10 },
    { 4 * _KB, 10 },
    { 16 * _KB, 10 },
    { 32 * _KB, 10 },
    { 64 * _KB, 10 },
    { 128 * _KB, 10 },
    { 192 * _KB, 10 },
    { 256 * _KB, 10 },
    { 384 * _KB, 10 },
    { 512 * _KB, 10 },
    { 1 * _MB, 10 },
    { 2 * _MB, 10 },
    { 4 * _MB, 10 },
    { 8 * _MB, 10 },
    { 16 * _MB, 3 },
    { 32 * _MB, 3 },
    { 64 * _MB, 3 },
};

INSTANTIATE_TEST_SUITE_P(mem_test,
                         latency_test,
                         ::testing::Combine(
                             ::testing::ValuesIn(sizes_and_iters),
                             ::testing::Values(2)),
                         PrintToStringParamName());
