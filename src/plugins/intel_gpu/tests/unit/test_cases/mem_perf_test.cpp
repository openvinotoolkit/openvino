// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "opencl_helper_instance.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>

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
