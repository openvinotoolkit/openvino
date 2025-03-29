// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ocl/ocl_wrapper.hpp>

static void checkStatus(int status, const char *message) {
    if (status != 0) {
        std::string str_message(message + std::string(": "));
        std::string str_number(std::to_string(status));

        throw std::runtime_error(str_message + str_number);
    }
}

struct OpenCL {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;
    std::shared_ptr<cl::UsmHelper> _usm_helper;
    bool _supports_usm;
    bool _out_of_order_queue;

    OpenCL(bool out_of_order_queue = true)
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
                        _out_of_order_queue = out_of_order_queue;

                        auto extensions = _device.getInfo<CL_DEVICE_EXTENSIONS>();
                        _supports_usm = extensions.find("cl_intel_unified_shared_memory") != std::string::npos;

                        _usm_helper = std::make_shared<cl::UsmHelper>(_context, _device, _supports_usm);

                        cl_command_queue_properties props = _out_of_order_queue ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
                        _queue = cl::CommandQueue(_context, _device, props);

                        return;
                    }
                }
            }
        }
    }

    OpenCL(cl::Device device, bool out_of_order_queue = true)
    {
        cl_uint n = 0;
        cl_int err = clGetPlatformIDs(0, NULL, &n);
        checkStatus(err, "clGetPlatformIDs");

        // Get platform list
        std::vector<cl_platform_id> platform_ids(n);
        err = clGetPlatformIDs(n, platform_ids.data(), NULL);
        checkStatus(err, "clGetPlatformIDs");

        _device = device;
        _context = cl::Context(_device);
        _out_of_order_queue = out_of_order_queue;

        auto extensions = _device.getInfo<CL_DEVICE_EXTENSIONS>();
        _supports_usm = extensions.find("cl_intel_unified_shared_memory") != std::string::npos;

        _usm_helper = std::make_shared<cl::UsmHelper>(_context, _device, _supports_usm);

        cl_command_queue_properties props = _out_of_order_queue ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
        _queue = cl::CommandQueue(_context, _device, props);
    }

    void releaseOclImage(std::shared_ptr<cl_mem> image) {
        checkStatus(clReleaseMemObject(*image), "clReleaseMemObject");
    }
};
