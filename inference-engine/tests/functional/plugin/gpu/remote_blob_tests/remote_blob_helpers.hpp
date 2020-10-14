// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#ifndef NOMINMAX
# define NOMINMAX
#endif

#ifdef _WIN32
# include <gpu/gpu_context_api_dx.hpp>
#elif defined ENABLE_LIBVA
# include <gpu/gpu_context_api_va.hpp>
#endif
#include <gpu/gpu_context_api_ocl.hpp>

struct OpenCL {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;

    explicit OpenCL(std::shared_ptr<std::vector<cl_context_properties>> media_api_context_properties = nullptr) {
        // get Intel iGPU OCL device, create context and queue
        {
            const unsigned int refVendorID = 0x8086;
            cl_uint n = 0;
            cl_int err = clGetPlatformIDs(0, NULL, &n);

            // Get platform list
            std::vector<cl_platform_id> platform_ids(n);
            err = clGetPlatformIDs(n, platform_ids.data(), NULL);

            for (auto& id : platform_ids) {
                cl::Platform platform = cl::Platform(id);
                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    if (refVendorID == d.getInfo<CL_DEVICE_VENDOR_ID>()) {
                        _device = d;
                        _context = cl::Context(_device);
                        break;
                    }
                }
            }
            cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            _queue = cl::CommandQueue(_context, _device, props);
        }
    }

    explicit OpenCL(cl_context context) {
        // user-supplied context handle
        _context = cl::Context(context, true);
        _device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);

        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        _queue = cl::CommandQueue(_context, _device, props);
    }
};
