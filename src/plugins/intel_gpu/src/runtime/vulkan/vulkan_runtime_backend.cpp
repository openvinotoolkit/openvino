// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime_backend_provider.hpp"

#include "openvino/core/except.hpp"
#include "vulkan_device_detector.hpp"
#include "vulkan_engine_factory.hpp"

namespace cldnn::backend_extensions {

const runtime_backend_descriptor& get_compiled_runtime_backend() {
    static const runtime_backend_descriptor descriptor{
        engine_types::vulkan,
        runtime_types::vulkan,
        "vulkan",
        runtime_interop_kind::native,
    };
    return descriptor;
}

std::map<std::string, std::shared_ptr<device>> query_compiled_runtime_devices(engine_types engine_type,
                                                                              runtime_types runtime_type,
                                                                              void* user_context,
                                                                              void* user_device,
                                                                              int context_device_id,
                                                                              int target_tile_id,
                                                                              bool initialize_devices) {
    OPENVINO_ASSERT(engine_type == engine_types::vulkan && runtime_type == runtime_types::vulkan);
    vulkan::vulkan_device_detector detector;
    return detector.get_available_devices(user_context,
                                          user_device,
                                          context_device_id,
                                          target_tile_id,
                                          initialize_devices);
}

std::shared_ptr<engine> create_compiled_runtime_engine(engine_types engine_type,
                                                       runtime_types runtime_type,
                                                       const std::shared_ptr<device>& device) {
    OPENVINO_ASSERT(engine_type == engine_types::vulkan && runtime_type == runtime_types::vulkan);
    return vulkan::create_vulkan_engine(device, runtime_type);
}

}  // namespace cldnn::backend_extensions
