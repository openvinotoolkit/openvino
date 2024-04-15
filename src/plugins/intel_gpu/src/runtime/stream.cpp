// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/stream.hpp"

#include "ocl/ocl_stream.hpp"

#include <stdexcept>

namespace cldnn {

QueueTypes stream::detect_queue_type(engine_types engine_type, void* queue_handle) {
    switch (engine_type) {
        case engine_types::sycl:
        case engine_types::ocl:
            return ocl::ocl_stream::detect_queue_type(queue_handle);
        default: throw std::runtime_error("Invalid engine type");
    }
}

}  // namespace cldnn
