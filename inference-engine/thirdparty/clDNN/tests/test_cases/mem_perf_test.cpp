// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include <cl2_wrapper.h>

static void checkStatus(int status, const char *message)
{
    if (status != 0)
    {
        std::string str_message(message + std::string(": "));
        std::string str_number(std::to_string(status));

        throw std::runtime_error(str_message + str_number);
    }
}


struct OpenCL
{
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;

    OpenCL()
    {
        // get Intel iGPU OCL device, create context and queue
        {
            static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";
            const uint32_t device_type = CL_DEVICE_TYPE_GPU;  // only gpu devices
            const uint32_t device_vendor = 0x8086;  // Intel vendor

            cl_uint n = 0;
            cl_int err = clGetPlatformIDs(0, NULL, &n);
            checkStatus(err, "clGetPlatformIDs");

            // Get platform list
            std::vector<cl_platform_id> platform_ids(n);
            err = clGetPlatformIDs(n, platform_ids.data(), NULL);
            checkStatus(err, "clGetPlatformIDs");

            for (auto& id : platform_ids) {
                cl::Platform platform = cl::Platform(id);

                auto vendor_id = platform.getInfo<CL_PLATFORM_VENDOR>();
                if (vendor_id != INTEL_PLATFORM_VENDOR)
                    continue;

                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    if (d.getInfo<CL_DEVICE_TYPE>() == device_type &&
                        d.getInfo<CL_DEVICE_VENDOR_ID>() == device_vendor) {
                        _device = d;
                        _context = cl::Context(_device);
                        return;
                    }
                }
            }
        }
    }
    void releaseOclImage(std::shared_ptr<cl_mem> image)
    {
        checkStatus(clReleaseMemObject(*image), "clReleaseMemObject");
    }
};

static size_t img_size = 400;
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
    std::cerr << "img_size=" << img_size << " iters=" << max_iter << " exec time: avg=" << avg << time_suffix << ", max=" << max << time_suffix << std::endl;
}

static void fill_input(uint8_t* ptr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        ptr[i] = static_cast<uint8_t>(i % 255);
    }
}

static void run_test(std::function<void()> body) {
    using Time = std::chrono::high_resolution_clock;
    int64_t max_time = 0;
    double avg_time = 0.0;
    for (size_t iter = 0; iter < max_iter; iter++) {
        auto start = Time::now();
        body();
        auto stop = Time::now();
        std::chrono::duration<float> fs = stop - start;
        time_interval d = std::chrono::duration_cast<time_interval>(fs);
        max_time = std::max(max_time, d.count());
        avg_time += static_cast<double>(d.count());
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


TEST(mem_perf_test, fill_input) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    cl::UsmMemory input_buffer(ctx);
    input_buffer.allocateHost(sizeof(uint8_t) * img_size * img_size);

    run_test([&]() {
        fill_input(static_cast<uint8_t*>(input_buffer.get()), img_size * img_size);
    });
}

TEST(mem_perf_test, buffer_no_lock) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::CommandQueue queue(ctx, device);

    run_test([&]() {
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });
}


TEST(mem_perf_test, buffer_lock_rw) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::CommandQueue queue(ctx, device);

    run_test([&]() {
        auto _mapped_ptr = queue.enqueueMapBuffer(input_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(uint8_t) * img_size * img_size, nullptr, nullptr);
        fill_input(static_cast<uint8_t*>(_mapped_ptr), img_size * img_size);
        queue.enqueueUnmapMemObject(input_buffer, _mapped_ptr);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });

    auto _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    validate_result(static_cast<float*>(_mapped_ptr), img_size * img_size);
    queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
}


TEST(mem_perf_test, buffer_lock_w) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::CommandQueue queue(ctx, device);

    run_test([&]() {
        auto _mapped_ptr = queue.enqueueMapBuffer(input_buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(uint8_t) * img_size * img_size, nullptr, nullptr);
        fill_input(static_cast<uint8_t*>(_mapped_ptr), img_size * img_size);
        queue.enqueueUnmapMemObject(input_buffer, _mapped_ptr);
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });

    auto _mapped_ptr = queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * img_size * img_size, nullptr, nullptr);
    validate_result(static_cast<float*>(_mapped_ptr), img_size * img_size);
    queue.enqueueUnmapMemObject(output_buffer, _mapped_ptr);
}

TEST(mem_perf_test, buffer_copy) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::Buffer input_buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * img_size * img_size);
    cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * img_size * img_size);
    cl::Kernel kernel(program, "simple_reorder");

    cl::CommandQueue queue(ctx, device);
    std::vector<uint8_t> input(img_size*img_size);

    run_test([&]() {
        fill_input(static_cast<uint8_t*>(input.data()), img_size * img_size);
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

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemcpyINTEL_fn)(
            cl_command_queue command_queue,
            cl_bool blocking,
            void* dst_ptr,
            const void* src_ptr,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

inline cl_int enqueue_memcpy(clEnqueueMemcpyINTEL_fn fn, const cl::CommandQueue& cpp_queue, void *dst_ptr, const void *src_ptr,
    size_t bytes_count, bool blocking = true, const std::vector<cl::Event>* wait_list = nullptr, cl::Event* ret_event = nullptr) {

    cl_event tmp;
    cl_int err = fn(
        cpp_queue.get(),
        static_cast<cl_bool>(blocking),
        dst_ptr,
        src_ptr,
        bytes_count,
        wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
        wait_list == nullptr ? nullptr : (cl_event*)&wait_list->front(),
        ret_event == nullptr ? nullptr : &tmp);

    if (ret_event != nullptr && err == CL_SUCCESS)
        *ret_event = tmp;

    return err;
}

TEST(mem_perf_test, usm_host) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;
    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::UsmMemory input_buffer(ctx);
    input_buffer.allocateHost(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory output_buffer(ctx);
    output_buffer.allocateDevice(device, sizeof(float) * img_size * img_size);
    cl::UsmMemory output_buffer_host(ctx);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);
    cl::Kernel kernel1(program, "simple_reorder");
    cl::KernelIntel kernel(kernel1, true);

    cl::CommandQueue queue(ctx, device);

    clEnqueueMemcpyINTEL_fn fn = load_entrypoint<clEnqueueMemcpyINTEL_fn>(queue.get(), "clEnqueueMemcpyINTEL");

    run_test([&]() {
        fill_input(static_cast<uint8_t*>(input_buffer.get()), img_size * img_size);
        kernel.setArgUsm(0, input_buffer);
        kernel.setArgUsm(1, output_buffer);
        cl::Event ev;
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(img_size*img_size), cl::NDRange(16), nullptr, &ev);
        cl::WaitForEvents({ev});
    });

    enqueue_memcpy(fn, queue,
                            output_buffer_host.get(),
                            output_buffer.get(),
                            sizeof(float) * img_size * img_size,
                            true,
                            nullptr,
                            nullptr);
    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}


TEST(mem_perf_test, usm_device_copy) {
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;
    cl::Program program(ctx, kernel_code);
    checkStatus(program.build({device}, ""), "build");
    cl::UsmMemory input_buffer(ctx);
    input_buffer.allocateHost(sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory input_buffer_device(ctx);
    input_buffer_device.allocateDevice(device, sizeof(uint8_t) * img_size * img_size);
    cl::UsmMemory output_buffer(ctx);
    output_buffer.allocateDevice(device, sizeof(float) * img_size * img_size);
    cl::UsmMemory output_buffer_host(ctx);
    output_buffer_host.allocateHost(sizeof(float) * img_size * img_size);
    cl::Kernel kernel1(program, "simple_reorder");
    cl::KernelIntel kernel(kernel1, true);

    cl::CommandQueue queue(ctx, device);
    clEnqueueMemcpyINTEL_fn fn = load_entrypoint<clEnqueueMemcpyINTEL_fn>(queue.get(), "clEnqueueMemcpyINTEL");

    run_test([&]() {
        cl::Event copy_ev;
        fill_input(static_cast<uint8_t*>(input_buffer.get()), img_size * img_size);
        cl::usm::enqueue_memcpy(queue,
                                input_buffer_device.get(),
                                input_buffer.get(),
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

    enqueue_memcpy(fn, queue,
                            output_buffer_host.get(),
                            output_buffer.get(),
                            sizeof(float) * img_size * img_size,
                            true,
                            nullptr,
                            nullptr);
    validate_result(static_cast<float*>(output_buffer_host.get()), img_size * img_size);
}
