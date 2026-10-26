// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "sycl_wrapper.hpp"
#include "sycl_device_detector.hpp"
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

#define SYCL_ERR_MSG_FMT(error) ("[GPU] " + std::string(error.what()) +  std::string(", SYCL error code: ") + std::to_string(error.code().value()))

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

template <typename HandlerT, typename FnT>
void host_task(HandlerT &cgh, const FnT &task) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    cgh.ext_codeplay_enqueue_native_command(task);
#else
    cgh.host_task(task);
#endif
}

}  // namespace sycl
}  // namespace cldnn
