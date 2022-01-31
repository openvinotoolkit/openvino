// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/stream.hpp"

#include "ocl/ocl_stream.hpp"
#ifdef GPU_ENABLE_ZE_BACKEND
#include "ze/ze_stream.hpp"
#endif //GPU_ENABLE_ZE_BACKEND

#include <stdexcept>

namespace cldnn {

queue_types stream::detect_queue_type(engine_types engine_type, void* queue_handle) {
    switch (engine_type) {
        case engine_types::ocl: return ocl::ocl_stream::detect_queue_type(queue_handle);
#ifdef GPU_ENABLE_ZE_BACKEND
        case engine_types::ze: return ze::ze_stream::detect_queue_type(queue_handle);
#endif //GPU_ENABLE_ZE_BACKEND
        default: throw std::runtime_error("Invalid engine type");
    }
}

}  // namespace cldnn
