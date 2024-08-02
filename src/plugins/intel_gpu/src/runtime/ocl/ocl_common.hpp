// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ocl_wrapper.hpp"
#include "ocl/ocl_device_detector.hpp"
#include "openvino/core/except.hpp"

#include <vector>

namespace cldnn {
namespace ocl {

using ocl_queue_type = cl::CommandQueue;
using ocl_kernel_type = cl::KernelIntel;

class ocl_error : public ov::Exception {
public:
    explicit ocl_error(cl::Error const& err);
};

#define OCL_ERR_MSG_FMT(error) ("[GPU] " + std::string(error.what()) + std::string(", error code: ") + std::to_string(error.err()))


/// WA: Force exit. Any opencl api call can be hang after CL_OUT_OF_RESOURCES.
inline void force_exit() {
    std::cerr << "[GPU] force exit.\n"
              << "\tDue to the driver bug any subsequent OpenCL API call will cause application hang, "
              << "so GPU plugin can't finish correctly.\n"
              << "\tPlease try to update the driver or reduce memory consumption "
              << "(use smaller batch size, less streams, lower precision, etc)"
              << "to avoid CL_OUT_OF_RESOURCES exception" << std::endl;
    std::_Exit(-1);
}

inline bool is_device_available(const device_info& info) {
    ocl_device_detector detector;
    auto devices = detector.get_available_devices(nullptr, nullptr);
    for (auto& device : devices) {
        if (device.second->get_info().uuid.uuid == info.uuid.uuid) {
            return true;
        }
    }

    return false;
}

inline void rethrow_or_exit(std::string message, cl_int error, const device_info& info) {
    if (error != CL_OUT_OF_RESOURCES) {
        OPENVINO_THROW(message);
    }
    // For CL_OUT_OF_RESOURCES exception there are 2 possible cases:
    // 1. Real out of resource which means that plugin must exit
    // 2. Device is lost during application run, plugin may throw an exception
    if (is_device_available(info)) {
        force_exit();
    } else {
        OPENVINO_THROW(message);
    }
}

inline void rethrow_or_exit(const cl::Error& error, const device_info& info) {
    rethrow_or_exit(OCL_ERR_MSG_FMT(error), error.err(), info);
}

}  // namespace ocl
}  // namespace cldnn
