// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "sycl_wrapper.hpp"
#include "sycl/sycl_device_detector.hpp"
#include "openvino/core/except.hpp"

#include <vector>

namespace cldnn {
namespace sycl {

using sycl_queue_type = ::sycl::queue;
using sycl_kernel_type = ::sycl::kernel;

class sycl_error : public ov::Exception {
public:
    explicit sycl_error(::sycl::exception const& err);
};

#define SYCL_ERR_MSG_FMT(error) ("[GPU] " + std::string(error.what()) +  std::string(", OpenCL error code: ") + std::to_string(error.code().value()))

inline bool is_device_available(const device_info& info) {
    sycl_device_detector detector;
    auto devices = detector.get_available_devices(nullptr, nullptr);
    for (auto& device : devices) {
        if (device.second->get_info().uuid.uuid == info.uuid.uuid) {
            return true;
        }
    }

    return false;
}

inline void rethrow(std::string message, cl_int error, const device_info& info) {
    if (error != CL_OUT_OF_RESOURCES) {
        OPENVINO_THROW(message);
    }
    // For CL_OUT_OF_RESOURCES exception there are 2 possible cases:
    // 1. Real out of resource
    // 2. Device is lost during application run, plugin may throw an exception
    if (is_device_available(info)) {
        std::stringstream ss;
        ss << "[GPU] CL_OUT_OF_RESOURCES exception.\n"
           << "\tDue to a driver bug, any subsequent OpenCL API call may cause the application to hang, "
           << "so the GPU plugin may be unable to finish correctly.\n"
           << "\tThe CL_OUT_OF_RESOURCES error typically occurs in two cases:\n"
           << "\t1. An actual lack of memory for the current inference.\n"
           << "\t2. An out-of-bounds access to GPU memory from a kernel.\n"
           << "\tFor case 1, you may try adjusting some model parameters (e.g., using a smaller batch size, lower inference precision, fewer streams, etc.)"
           << " to reduce the required memory size. For case 2, please submit a bug report to the OpenVINO team.\n"
           << "\tAdditionally, please try updating the driver to the latest version.\n";
        OPENVINO_THROW(ss.str());
    } else {
        OPENVINO_THROW(message);
    }
}

inline void rethrow(const ::sycl::exception& error, const device_info& info) {
    rethrow(SYCL_ERR_MSG_FMT(error), error.code().value(), info);
}

}  // namespace sycl
}  // namespace cldnn
