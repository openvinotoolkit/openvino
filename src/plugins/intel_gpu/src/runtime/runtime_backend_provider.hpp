// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/runtime_backend_registry.hpp"

namespace cldnn::backend_extensions {

/// Runtime implementation supplied by the optional backend compiled into the plugin.
const runtime_backend_descriptor& get_compiled_runtime_backend();

std::map<std::string, std::shared_ptr<device>> query_compiled_runtime_devices(engine_types engine_type,
                                                                              runtime_types runtime_type,
                                                                              void* user_context,
                                                                              void* user_device,
                                                                              int context_device_id,
                                                                              int target_tile_id,
                                                                              bool initialize_devices);

std::shared_ptr<engine> create_compiled_runtime_engine(engine_types engine_type,
                                                       runtime_types runtime_type,
                                                       const std::shared_ptr<device>& device);

}  // namespace cldnn::backend_extensions
