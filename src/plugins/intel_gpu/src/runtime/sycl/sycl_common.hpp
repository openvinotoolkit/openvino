// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include <algorithm>
#include <vector>

#include "openvino/core/except.hpp"
#include "sycl_device_detector.hpp"
#include "sycl_wrapper.hpp"

namespace cldnn {
namespace sycl {

using sycl_queue_type = ::sycl::queue;
using sycl_kernel_type = ::sycl::kernel;

class sycl_error : public ov::Exception {
public:
    explicit sycl_error(::sycl::exception const& err);
};

#define SYCL_ERR_MSG_FMT(error) ("[GPU] " + std::string(error.what()) +  std::string(", SYCL error code: ") + std::to_string(error.code().value()))

inline bool is_device_available(const device_info& info) {
    auto devices = sycl_device_detector::get_available_devices(nullptr, nullptr);
    return std::any_of(devices.begin(), devices.end(), [&info](const auto& device) {
        return device.second->get_info().uuid.uuid == info.uuid.uuid;
    });
}

}  // namespace sycl
}  // namespace cldnn
