/*
// Copyright (c) 2018 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "ocl_builder.h"
#include "confiugration.h"

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace cldnn { namespace gpu{

    ocl_builder::ocl_builder(const configuration& config)
        : _is_user_context(config.user_context != nullptr ? true : false)
    {
        if (_is_user_context)
        {
            _context = *config.user_context;
            build_device_from_user_context(config);
        }
        else
        {
            build_device(config);
            build_context();
        }
        build_platform_id();
    }

    void ocl_builder::build_device_from_user_context(const configuration& config)
    {
        auto all_devices = _context.getInfo<CL_CONTEXT_DEVICES>();
        auto num_devices = _context.getInfo<CL_CONTEXT_NUM_DEVICES>();
        if (num_devices != 1)
        {
            throw std::runtime_error("[ERROR]. Number of devices from user context is not equal to 1.");
        }
        auto device = all_devices.at(0);
        auto dev_type = device.getInfo<CL_DEVICE_TYPE>();
        if (dev_type != CL_DEVICE_TYPE_GPU)
        {
            throw std::runtime_error("[ERROR]. User defined device is not an gpu device!");
        }

        std::list<std::string> reasons;
        if (does_device_match_config(config, device, reasons))
        {
            _device = device;
            return;
        }
        else
        {
            std::string error_msg = "No OpenCL device found which would match provided configuration:";
            for (const auto& reason : reasons)
                error_msg += "\n    " + reason;
            throw std::invalid_argument(std::move(error_msg));
        }

    }

    void ocl_builder::build_device(const configuration& config)
    {
        std::list<std::string> reasons;
        cl_uint n = 0;

        // Get number of platforms availible
        cl_int err = clGetPlatformIDs(0, NULL, &n);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("clGetPlatformIDs error " + std::to_string(err));
        }

        // Get platform list
        std::vector<cl_platform_id> platform_ids(n);
        err = clGetPlatformIDs(n, platform_ids.data(), NULL);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("clGetPlatformIDs error " + std::to_string(err));
        }

        for (auto& id : platform_ids)
        {
            cl::Platform platform = cl::Platform(id);
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (auto& d : devices)
            {
                if (does_device_match_config(config, d, reasons))
                {
                    _device = d;
                    return;
                }
            }
        }

        if (reasons.empty())
            throw std::runtime_error("Could not find any OpenCL device");

        std::string error_msg = "No OpenCL device found which would match provided configuration:";
        for (const auto& reason : reasons)
            error_msg += "\n    " + reason;

        throw std::invalid_argument(std::move(error_msg));
    }

    void ocl_builder::build_context()
    {
        _context = cl::Context(_device);
    }

    bool ocl_builder::does_device_match_config(const configuration& config, const cl::Device& dev, std::list<std::string>& reasons)
    {
        auto dev_name = dev.getInfo<CL_DEVICE_NAME>();
        bool ok = true;

        auto dev_type = dev.getInfo<CL_DEVICE_TYPE>();

        cl_device_type device_types[] = {
            CL_DEVICE_TYPE_DEFAULT,
            CL_DEVICE_TYPE_CPU,
            CL_DEVICE_TYPE_GPU,
            CL_DEVICE_TYPE_ACCELERATOR };

        if (dev_type != device_types[config.device_type])
        {
            reasons.push_back(dev_name + ": invalid device type");
            ok = false;
        }

        auto vendor_id = dev.getInfo<CL_DEVICE_VENDOR_ID>();
        if (vendor_id != config.device_vendor)
        {
            reasons.push_back(dev_name + ": invalid vendor type");
            ok = false;
        }

        if (config.host_out_of_order)
        {
            auto queue_properties = dev.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
            using cmp_t = std::common_type<decltype(queue_properties), typename std::underlying_type<cl::QueueProperties>::type>::type;
            if (!(static_cast<cmp_t>(queue_properties) & static_cast<cmp_t>(cl::QueueProperties::OutOfOrder)))
            {
                reasons.push_back(dev_name + ": missing out of order support");
                ok = false;
            }
        }
        return ok;
    }

    void ocl_builder::build_platform_id()
    {
        cl_int err;
        _platform_id = _device.getInfo<CL_DEVICE_PLATFORM>(&err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Error getting OpenCL platform_id from device!");
        }
    }

}
}

